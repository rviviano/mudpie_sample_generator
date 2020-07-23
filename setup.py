#!/usr/bin/env python

from setuptools import setup, find_packages

def get_version():
    with open("mudpie_sample_generator.py", 'r') as f:
        for line in f:
            if "__version__" in line:
                return line.split("=")[1].strip()
           

setup(name='mudpie-sample-generator',
      version=get_version(),
      description='Extract random samples from wav file and save them.',
      author='Raymond Viviano',
      author_email='rayviviano@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      entry_points = { 
            'console_scripts': [ 
                'mudsampgen = mudpie_sample_generator.py:main'
            ] 
        }, 
      license='LICENSE.md',
    )
