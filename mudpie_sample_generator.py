#!/usr/bin/env python3

# Mudpie Sample Generator
#
# A script to automate sample creation from a mudpie. Randomly selects 
# .3 to 3 second chunks of audio. Bandpass filters, applies a random amplitude 
# envelope to, normalizes, gates, demeans, detrends and declicks the samples.
#
# Raymond Viviano
# July 20th, 2020
# rayvivianomusic@gmail.com

# Dependencies - TODO: Not sure if all of these will be needed yet.
from __future__ import print_function, division
import os, sys, getopt, traceback, functools, warnings
import wave, sndhdr, wavio
import numpy as np 
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join, basename, splitext, exists
from copy import deepcopy
from scipy.signal import detrend, butter, sosfiltfilt, savgol_filter
from scipy.interpolate import BSpline, splrep
from numpy.random import randint

# No real rhyme or reason to this
__version__ = "0.1.0"

# TODO: Implement tests
# TODO: Implement embarrassingly parallel processing
# NOTE: Because of the demean/detrend step, this script probably doesn't work 
#       well with rectified waveforms

# Class Definitions
class WavError(Exception):
    pass

# Decorator definitions
def deprecated(func):
    """Decorator to mark deprecated functions. Emits a warning on function call."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)
    return new_func


# Function Definitions
def process_options():
    """
        Process command line arguments for file input, output, and number of 
        samples. Also provide an option for help. Return input file, ouput dir, 
        and output prefix. Print usage and exit if help requested.
    """

    # Define usage
    usage = """

    Usage: python mudpie-sample-generator.py --in <arg> --out <arg> --pre <arg> 
                                             --num <arg> -h

    Mandatory Options:

        --i, --in      The mudpie, must be 16- or 24-bit int wav. No 32float 
                       yet. Also, it needs to be *much* longer than 10 seconds 
                       in length; otherwise, there is not much point in running
                       this script.

        --o, --out     Directory to save output wavs to. If the provided 
                       directory path does not exist, then the script will try 
                       to create it.

        --p, --pre     Filename prefix for all output wav samples.

        --n, --num     Number of samples to generate (it maxes out at 400 for 
                       your own good). 

    Optional Options:

        -h             Print this usage message and exit

    """

    # Get commandline options and arguments
    opts, _ = getopt.getopt(sys.argv[1:], "h:", ["in=", "out=", "pre=", "num="])

    # Set variables to defaults
    in_path, out_dir, prefix, num_samps = None, None, None, 1

    for opt, arg in opts:
        # Mandatory arguments
        if opt == "--i" or opt == "--in":
            if (arg is not None) and (isfile(abspath(arg))):
                # Check that the file is a wav
                try:
                    wv_hdr = sndhdr.what(abspath(arg))
                    if wv_hdr is not None:
                        in_path = arg
                    else: 
                        raise WavError("You must supply a 16- or 24-bit int wav.")
                except WavError:
                    raise
                except:
                    traceback.print_exc(file=sys.stdout)
                    print(os.linesep + 'Unexpected input error.')
                    sys.exit(1)

        if opt == "--o" or opt == "--out": 
            if arg is not None:
                out_dir = arg
                # If the specified out dir does not exist, try to make it
                if not isdir(abspath(out_dir)): 
                    try:
                        os.makedirs(abspath(out_dir))
                    except:
                        traceback.print_exc(file=sys.stdout)
                        print(os.linesep + 'Cannot create output directory.')
                        sys.exit(1)

        if opt == "--p" or opt == "--pre":
            if arg is not None:
                prefix = arg
            else:
                prefix = "randSamp"

        if opt == "--n" or opt == "--num":
            if arg is not None:
                num_samps = max(1, int(arg))
                num_samps = min(num_samps, 400)
            else:
                num_samps = 1

        # Optional options
        if opt == "-h":
            print(usage)
            sys.exit(0)


    # Make sure that arguments exist for all mandatory options
    if None in [in_path, out_dir, prefix, num_samps]:
        print(os.linesep + 'Errors detected with mandatory options')
        print(usage)
        sys.exit(1)

    # Return options for audio processing
    return in_path, out_dir, prefix, num_samps


def load_wav(wave_filepath):
    """Load wav data into np array and also return important wav parameters."""
    wv = wavio.read(wave_filepath)
    wav_data = wv.data 
    num_frames = wav_data.shape[0]
    framerate = wv.rate
    samplewidth = wv.sampwidth                                   
    return  wav_data, framerate, num_frames, samplewidth


def check_wav(framerate, num_frames):
    # TODO: Reject 8-bit samples
    sample_length = convert_frames_to_ms(num_frames, framerate)
    if sample_length < 10000:
        raise WavError("Input wav needs to be longer than 10 seconds")


def convert_ms_to_frames(sample_length, framerate):
    """Convert sample length in milleconds to number of frames."""
    return int(framerate * (sample_length/1000.0))


def convert_frames_to_ms(num_frames, framerate):
    """Convert number of frames to sample length in milleconds."""
    return int(num_frames * 1000.0/framerate)


def random_mudpie_sample(wav_data, sample_length, framerate, count=0):
    """
    Extract a random sample of audio from the wav data that is as long
    as the provided sample length. Check if the sample starts with silence
    or contains too much silence after applying a noise gate. If too silent,
    this function calls itself recursively.

    Inputs:

        wav_data:       numpy array of complete mudpie data

        sample_length:  This is the sample length in ms

        framerate:      Audio sample rate. Usually 44.1 or 48 khz.

        count:          If this hits 100, the provided wav may have too much 
                        silence and the recursion may not break on its own. 
                        Raise a WavError if this occurs.
        
    Output:

        sample:         Random sample from the provided mudpie to process more.    
    """

    # Increment the counter
    count += 1

    # Check if there have been 100 calls. If so, terminate in error.
    if count >= 100:
        msg = ("There is too much silence in the waveform to easily " +
               "extract meaningful samples.")
        raise WavError(msg)

    # Get the sample_length of the random sample in frames
    sample_frames = convert_ms_to_frames(sample_length, framerate)

    # Get a random index to the wav_data array to slice at. Make sure the 
    # index plus sample_frames don't exceed the last element of the array 
    slice_idx = randint(0, wav_data.shape[0] - sample_frames - 1)

    # Extract the random sample
    sample = wav_data[slice_idx:slice_idx+sample_frames, :]

    # Gate sample, attempt to remove noise from "silent" sections
    gated_samp = gate_audio(sample, framerate, wav_data.dtype)

    # TODO: The gate function might not be good enough to test in this way.
    # UPDATE: Tweaked the gate params to make it faster. Maybe it will work
    # better now???
    # Make sure there is audio within the first 10 ms of the sample
    frames_10ms = convert_ms_to_frames(10, framerate)
    sample_10_ms = gated_samp[:frames_10ms, :]
    if 0 in sample_10_ms:
        sample = random_mudpie_sample(wav_data, sample_length, framerate, count)
    
    # TODO: The gate function might not be good enough for this
    # Make sure the the first 50 ms contains 75% audio
    frames_50ms = convert_ms_to_frames(50, framerate)
    sample_50_ms = gated_samp[:frames_50ms, :]
    if np.count_nonzero(sample_50_ms) < np.sum(sample_50_ms.shape)*.25:
        sample = random_mudpie_sample(wav_data, sample_length, framerate, count)

    # Make sure >50% of the sample isn't silence
    if np.count_nonzero(gated_samp) < sample.shape[0]*sample.shape[1]/2:
        sample = random_mudpie_sample(wav_data, sample_length, framerate, count)

    return sample
  

def bandpass_sample(sample, framerate):
    """ Determine bandbass filter second-order-sections for the provided 
        framerate and then return the sample after applying the filter.
    """
    # Convenience parameters
    nyquist = framerate * 0.5
    lowcut = 28.0/nyquist
    highcut = 18000.0/nyquist
    # Define the filter
    sos = butter(2, [lowcut, highcut], btype='bandpass', output='sos')
    # Bandpass the audio
    proc_sample = sosfiltfilt(sos, sample, axis=0)
    return proc_sample


def demean(sample):
    """Center waveform at 0 and maintain the input array's original type."""
    dtype = sample.dtype
    if len(sample.shape) > 1:
        # Get the mean of each channel separately
        sample_mean = sample.mean(axis=0).astype(dtype)
        sample -= sample_mean[np.newaxis, :]
        sample = sample.astype(dtype, casting="unsafe")
    else:
        # Data with only one axis supplied
        sample -= sample.mean().astype(dtype)
        sample = sample.astype(dtype, casting="unsafe")

    return sample


def generate_spline_curve(ctrls_x, ctrls_y, start, stop, n):
    # Find the knots and coefficients of 1-D curve B-spline representation 
    t, c, k = splrep(ctrls_x, ctrls_y, s=2, k=3)
    # Create B-spline object
    spline = BSpline(t, c, k, extrapolate=False)
    # Return the curve
    return spline(np.linspace(start, stop, n))


def smooth_hann(x):
    """Convolve signal (or envelope segments) with scaled window to smooth"""
    window_len = 200
    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]] # pad with reflected copies of signal
    w = np.hanning(window_len)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(int(window_len/2)-1):-(int(window_len/2))] # remove the padding when returning the convolved signal


def generate_amp_envelope(num_frames):
    """ Generate a random amplitude envelope to shape sample dynamics."""
    # Set first inflection point between 1/16 and 1/2 sample length 
    first_inflect_x = randint(num_frames*.0625, num_frames*.5)
    first_inflect_y = 1.0
    # Set second inflection between first inflection and ~3/4 sample length
    second_inflect_x = randint(first_inflect_x+50, num_frames*.75+10)
    second_inflect_y = randint(250, 750)*.001

    # Attack - Random, monotonically increasing control points
    ctrls_x = np.sort(randint(1, first_inflect_x-1, 50))
    ctrls_y = np.sort(randint(1, 999, 50)*.001)
    ctrls_x[0], ctrls_y[0] = (0, 0)
    ctrls_x[-1], ctrls_y[-1] = (first_inflect_x, first_inflect_y)
    
    attack = generate_spline_curve(ctrls_x, ctrls_y, 0, first_inflect_x, 
                                   first_inflect_x)

    # Sustain/Decay - same method as for the attack, but monotonic decrease
    ctrls_x = np.sort(randint(1, second_inflect_x-first_inflect_x-1, 30))
    ctrls_y = np.sort(randint(int(second_inflect_y*1000)-1, 999, 30)*.001)[::-1]
    ctrls_x[0], ctrls_y[0] = (0, 1.0)
    ctrls_x[-1] = second_inflect_x - first_inflect_x
    ctrls_y[-1] = second_inflect_y

    sus_decay = generate_spline_curve(ctrls_x, ctrls_y, 0, 
                                      second_inflect_x - first_inflect_x, 
                                      second_inflect_x - first_inflect_x)
    
    # Release - same method as sustain/decay

    ctrls_x = np.sort(randint(1, num_frames-second_inflect_x-1, 30))
    ctrls_y = np.sort(randint(0, int(second_inflect_y*1000)-1, 30)*.001)[::-1]
    ctrls_x[0], ctrls_y[0] = (0, second_inflect_y)
    ctrls_x[-1], ctrls_y[-1] = (num_frames - second_inflect_x, 0)

    release = generate_spline_curve(ctrls_x, ctrls_y, 0, 
                                    num_frames - second_inflect_x, 
                                    num_frames - second_inflect_x)

    envelope = np.concatenate((attack, sus_decay, release))

    # Set first and last 100 frames to 0 before smoothing
    envelope[:100] = np.zeros(100)
    envelope[-100:] = np.zeros(100)

    # Smooth to get rid of harsh edges between random curves
    envelope = smooth_hann(envelope)

    # Clip the vector (make sure everything is between 0 and 1)
    envelope = np.clip(envelope, 0, 1)

    return envelope[:, np.newaxis]


def apply_random_amp_envelope(sample):
    """
        Apply a random amplitude envelope to the sample
    """
    envelope = generate_amp_envelope(sample.shape[0])
    proc_sample = np.multiply(sample, envelope)
    return proc_sample


def gate_audio(sample, framerate, raw_dtype):
    """
        Simple gate to try to get rid of the noise floor in the output audio.
        Vectorizing this seems non-trivial. It might be one of the slowest 
        parts of the script. TODO: Time this function and compare it to
        other functions. This might be the first place to optimize.
    """
    attack = 0.4
    release = 0.85
    envelope = 0.0
    gain = 1.0

    # Set threshold based on in/output bitdepth
    if raw_dtype == np.int16:
        threshold = 1500.0 # TODO: This is still an untested threshold
    else: # 24-bit
        threshold = 15000.0

    for i in range(0, sample.shape[0]-1):
        # Calculate maxium absolute values across channels and cast to 
        # a native python float so that it matches the envelope*release type
        frame_max = float(np.max(np.abs(sample[i, :])))

        # Calculate signal envelope at frame 
        envelope = max(frame_max, envelope*release)

        # If the envelope is below the threshold, target gain is zero
        if envelope < threshold:
            target_gain = 0.0
        else:
            target_gain = 1.0

        # However, the actual gain applied by the limiter depends on the attack
        # This ensures a smooth onset of the noise gate
        gain = gain*attack + target_gain*(1-attack)
        # If gain is < 1e-8, it's effectivly 0. Cap gain at 1. No boosting.
        if gain < 0.00000001:
            gain = 0.0
        elif gain > 1.0:
            gain = 1.0

        # Apply potential gain reduction to the frame 
        sample[i, :] = gain * sample[i, :]
    
    return sample


def normalize_sample(sample, dtype):
    """Maximize sample amplitude."""

    if dtype == np.int16:
        ratio = 32767/np.max(np.abs(sample))
    elif dtype == np.int32:  # 24 bit wav casts to 32 bit np array 
        ratio = 8388607/np.max(np.abs(sample)) # Treat max as 24 int signed limit
    
    normalized_sample = ratio * sample

    return normalized_sample


def declick_sample(sample, framerate):
    """Ramp up and ramp down the signal over the first and last 20 ms."""
    num_frames = convert_ms_to_frames(20, framerate)
    ramp_up = np.linspace(0,1,num_frames)[:,np.newaxis]
    ramp_down = np.linspace(1,0,num_frames)[:,np.newaxis]
    sample_start = sample[0:num_frames, :]
    sample_end = sample[sample.shape[0]-num_frames:, :] 
    sample[0:num_frames, :] = np.multiply(sample_start, ramp_up)
    sample[sample.shape[0]-num_frames:, :] = np.multiply(sample_end, ramp_down)
    return sample


def main():
    """
        Load wav, check for issues, and pull random samples from the wav. 
        Bandpass filters, gates, applies a random amplitude envelope, 
        normalizes, detrends (possibly unnecessary), and declicks the samples
        before writing them to wav. 

        This script primarily works with audio
    """
    # Get input wav filepath and output specifications
    in_path, out_dir, prefix, num_samps = process_options()

    # Load the wav file and params
    wav_data, framerate, num_frames, samplewidth = load_wav(in_path)

    # Make sure the wav is long enough (> 10 seconds)
    check_wav(framerate, num_frames)

    # Generate random sample lengths between 250 and 3000 milliseconds
    sample_lengths = randint(300, 3001, num_samps)

    # Generate random sample
    for idx, sample_length in enumerate(sample_lengths):
        # Random sample output filepath
        zeropad_idx = "0"*(3 - len(str(idx+1))) + str(idx+1)
        out_path = join(abspath(out_dir), prefix + "_" + zeropad_idx + ".wav")

        # Extract a random sample from the mudpie
        raw_sample = random_mudpie_sample(wav_data, sample_length, framerate)

        # Bandpass the audio
        proc_sample = bandpass_sample(raw_sample, framerate)

        # Gate the Audio
        proc_sample = gate_audio(proc_sample, framerate, raw_sample.dtype)

        # Generate randomish amplitude envelope and apply to audio
        proc_sample = apply_random_amp_envelope(proc_sample)

        # Demean and detrend audio
        proc_sample = demean(proc_sample)
        proc_sample = detrend(proc_sample, axis=0)

        # Normalize the audio
        proc_sample = normalize_sample(proc_sample, raw_sample.dtype)
        
        # Declick the audio
        proc_sample = declick_sample(proc_sample, framerate)

        # Write output to a wav file. The audio processing occured with 64bit 
        # float precision. But will now typecast down to the original audio 
        # type during write.
        wavio.write(out_path, proc_sample, scale="none", rate=framerate, sampwidth=samplewidth)

        # # Uncomment to compare filtering, etc. with raw random sample
        # raw_path = join(abspath(out_dir), + prefix + "_raw_" + 
        #                 zeropad_idx + ".wav")            
        # wavio.write(raw_path, raw_sample, scale="none", rate=framerate, 
        #             sampwidth=samplewidth)


if __name__ == "__main__":
    main()