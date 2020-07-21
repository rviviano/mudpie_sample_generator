# Mudpie Sample Generator
#
# A script to automate sample creation from a mudpie. Randomly selects 
# .5 to 4 second chunks of audio. Bandpass filters, compresses, normalizes, 
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
import scipy.signal
import multiprocessing as mp
from os.path import isdir, isfile, abspath, join, basename, splitext, exists
from copy import deepcopy

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

        --i, --in      The mudpie, must be 16- or 24-bit int wav. No 32float yet

        --o, --out     Directory to save output wavs to

        --p, --pre     Filename prefix for all output wav samples

        --n, --num     Number of samples to generate (max 400 for your own good) 

    Optional Options:

        -h             Print this usage message and exit

    """

    # Get commandline options and arguments
    opts, _ = getopt.getopt(sys.argv[1:], "h:", ["in=", "out=", "pre=", "num="])

    # Set variables to defaults
    in_path, out_path, out_prefix, num_samps = None, None, None, 1

    for opt, arg in opts:
        # Mandatory arguments
        if opt == "--i" or opt == "--in":
            if (arg is not None) and (isfile(abspath(arg))):
                # Check that the file is a wav
                try:
                    wv_hdr = sndhdr.what(join(input_dir, f))
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
                out_path = arg
                # If the specified out dir does not exist, try to make it
                if not isdir(abspath(out_path)): 
                    try:
                        os.makedirs(abspath(out_path))
                    except:
                        traceback.print_exc(file=sys.stdout)
                        print(os.linesep + 'Cannot create output directory.')
                        sys.exit(1)

        if opt == "--p" or opt == "--pre":
            if arg is not None:
                out_prefix = arg
            else:
                out_prefix = "randSamp"

        if opt == "--n" or opt == "--num":
            if arg is not None:
                num_samps = max(1, arg)
                num_samps = min(num_samps, 400)
            else:
                num_samps = 1

        # Optional options
        if opt == "-h":
            print(usage)
            sys.exit(0)


    # Make sure that arguments exist for all mandatory options
    if None in [in_path, out_path, out_prefix, num_samps]:
        print(os.linesep + 'Errors detected with mandatory options')
        print(usage)
        sys.exit(1)

    # Return options for audio processing
    return in_path, out_path, out_prefix, num_samps


def load_wav(wave_filepath):
    # TODO: Determine if I should scrap this function or extend it                                  
    return wavio.read(wave_filepath)   


def normalize_sample(sample):
    """Maximize sample amplitude."""
    if sample.dtype = np.int16:
        normalized_sample = sample * 32767/np.max(np.abs(sample))
    elif sample.dtype = np.int32:  # 24 bit wav casts to 32 bit np array 
        normalized_sample = sample * 8,388,607/np.max(np.abs(sample))

    # TODO: Check that type is maintained
    return normalized_sample
        

def convert_ms_to_frames(sample_length, framerate):
    """Convert sample length in milleconds to number of frames."""
    return int(framerate * (sample_length/1000.0))


def convert_frames_to_ms(num_frames, framerate):
    """Convert number of frames to sample length in milleconds."""
    return int(num_frames * 1000.0/framerate)


def declick_sample(sample, framerate):
    pass


def process_wav():
    # TODO
    pass


def main():
    """
        TODO: Write this docstring
    """
    # Get input wav filepath and output specifications
    in_path, out_path, out_prefix, num_samps = process_options()

    # Load the wav file
    wv = load_wav(in_path)

    print(dir(wv))

    # Write output to a wav file
    # wavio.write(outfile, wv2, rate=framerate, sampwidth=samplewidth)


if __name__ == "__main__":
    main()