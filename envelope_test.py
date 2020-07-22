# Amplitude envelope test.
#
# Basically plots out randomly generated envelopes to test
# if the spline-generating code works as expected.
#
# Raymond Viviano
# July 22nd, 2020

import os, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from os.path import abspath, realpath, dirname

# Import mudpie_sample_generator to access random envelope generation
scriptpath = abspath(dirname(realpath(__file__)))
sys.path.append(scriptpath)
from mudpie_sample_generator import generate_amp_envelope


def plot_amp_envelope():
    x = np.linspace(0, 13230, 13231)[:,np.newaxis]
    y = generate_amp_envelope(13231)
    plt.plot(x,y)
    plt.show()


def main():
    for i in range(10):
        plot_amp_envelope()


if __name__ == "__main__":
    main()