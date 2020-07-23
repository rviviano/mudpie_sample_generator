# Mudpie Sample Generator
 Automatically create a bunch of processed audio samples from a mudpie 

 A better name might have been "Mudpie Slicer" as the script doesn't generate mangled waveforms for you. Instead, the script takes an input wav file, pulls random samples between 300 and 3000 milliseconds, processes them, and then saves them as individual samples in the directory of your choosing. 

 Processing includes: bandpass filtering(20 Hz and 18 kHz), noise gating, random amplitude envelope generation, demeaning, detrending, normalization, and declicking.

## Getting Started
 These instructions will get you a copy of this project up and running on your system.

 ### Prerequisites 
 * python https://www.python.org/
 * numpy  https://numpy.org/
 * scipy  https://www.scipy.org/
 * wavio  https://pypi.org/project/wavio/

 Tested with python 3.8 on Windows 10

 ### Installation 
 From the command line (Mac/Linux Terminal or Windows PowerShell).
 
 ```
 pythom -m pip install git+https://github.com/rviviano/mudpie-sample-generator.git@v0.1.0
 ```
 
## Usage
 mudpie_sample_generator.py is a commandline tool

 ```
 python mudpie_sample_generator.py --i <path to input wav> --o <directory to save output> --p <filename prefix for output samples> --n <number of samples to generate>
 ```

## Authors
 * **Raymond Viviano** - *Initial Work* - https://github.com/rviviano

## License
 This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
 * **ill.Gates** - For coining the term "mudpie" in a sound design context
 * **Mr.Bill** - For popularizing the term "mudpie" in music production

 If the origin of the term goes further back then that though, please let me know. 
