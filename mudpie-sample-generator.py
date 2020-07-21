# Mudpie Sample Generator
#
# A script to automate sample creation from a mudpie. Randomly selects 
# .25 to 3 second chunks of audio. Bandpass filters, compresses, normalizes, 
# applies a random amplitude envelope, and declicks.
#
# Raymond Viviano
# July 20th, 2020
# rayvivianomusic@gmail.com

# Dependencies - TODO: Not sure if all of these are needed yet.
from __future__ import print_function, division
import os, sys, getopt, traceback
import wave, sndhdr, wavio
import numpy as np 
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join, basename, splitext, exists
from copy import deepcopy
from scipy.signal import resample, detrend, butter, sosfiltfilt

# Meh, updated whenever I feel like it
__version__ = "0.0.1"

# TODO: Implement test suite

# Class Definitions
class WavError(Exception):
    pass


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
    sample_length = convert_frames_to_ms(num_frames, framerate)
    if sample_length < 10000:
        raise WavError("Input wav needs to be longer than 10 seconds")


def normalize_sample(sample, raw_dtype):
    """Maximize sample amplitude."""

    if raw_dtype == np.int16:
        ratio = 32767/np.max(np.abs(sample))
    elif raw_dtype == np.int32:  # 24 bit wav casts to 32 bit np array 
        ratio = 8388607/np.max(np.abs(sample))
    
    normalized_sample = ratio * sample

    return normalized_sample


def random_mudpie_sample(wav_data, sample_length, framerate, count=0):
    """
    Extract a random sample of audio from the wav data that is as long
    as the provided sample length.

    Inputs:

        wav_data:       numpy array of mudpie data

        sample_length:  This is the sample length in ms

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

    # Get a random index to the wav_data array to slice at. Make sure that 
    # the index plus the sample_frames do not exceed the last element of the 
    # array for any out-of-bounds error.
    slice_idx = np.random.randint(0, wav_data.shape[0] - sample_frames - 1)

    # Extract the random sample
    sample = wav_data[slice_idx:slice_idx+sample_frames, :]

    # Make sure more than half the sample isn't a bunch of 0s. Else, recursion.
    if np.count_nonzero(sample) < sample.shape[0]*sample.shape[1]/2:
        sample = random_mudpie_sample(wav_data, sample_length, count)
    
    return sample
  

def convert_ms_to_frames(sample_length, framerate):
    """Convert sample length in milleconds to number of frames."""
    return int(framerate * (sample_length/1000.0))


def convert_frames_to_ms(num_frames, framerate):
    """Convert number of frames to sample length in milleconds."""
    return int(num_frames * 1000.0/framerate)


def bandpass_sample(sample, framerate):
    """ Determine bandbass filter coefficients for the provided 
        framerate and then return the sample after applying the filter
    """
    nyquist = framerate * 0.5
    lowcut = 60.0/nyquist
    highcut = 12500.0/nyquist 
    # Bandpass the audio
    sos = butter(1, [lowcut, highcut], btype='bandpass', output='sos')
    proc_sample = sosfiltfilt(sos, sample, axis=0)
    # Not sure if this step is necessary
    proc_sample = detrend(proc_sample, axis=0)
    return proc_sample


def declick_sample(sample, framerate):
    """Ramp up and ramp down the signal over the first and last 20 ms"""
    pass


def main():
    """
        TODO: Write this docstring
    """
    # Get input wav filepath and output specifications
    in_path, out_dir, prefix, num_samps = process_options()

    # Load the wav file and params
    wav_data, framerate, num_frames, samplewidth = load_wav(in_path)

    # Generate random sample lengths between 250 and 3000 milliseconds
    sample_lengths = np.random.randint(250, 3001, num_samps)

    # Generate random sample
    for idx, sample_length in enumerate(sample_lengths):
        # Random sample output filepath
        zeropad_idx = "0"*(3 - len(str(idx+1))) + str(idx+1)
        out_path = join(abspath(out_dir), prefix + "_" + zeropad_idx + ".wav")

        # Extract a random sample from the mudpie
        raw_sample = random_mudpie_sample(wav_data, sample_length, framerate)

        # Bandpass the audio
        proc_sample = bandpass_sample(raw_sample, framerate)

        # Normalize the audio
        proc_sample = normalize_sample(proc_sample, raw_sample.dtype)

        # Declick the audio

        # Cast np array to either 16bit int or 24bit int.
        proc_sample = proc_sample.astype(raw_sample.dtype, casting="unsafe")

        # Write output to a wav file
        wavio.write(out_path, proc_sample, rate=framerate, sampwidth=samplewidth)

        # Temporary for comparing filtering, etc. with raw random sample
        raw_path = join(abspath(out_dir), "_raw_" + zeropad_idx + ".wav")
        wavio.write(raw_path, raw_sample, rate=framerate, sampwidth=samplewidth)

if __name__ == "__main__":
    main()