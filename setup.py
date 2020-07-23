#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mudipie-sample-generator',
      version='0.1.0',
      description='Extract random samples from wav file and save them.',
      author='Raymond Viviano',
      author_email='rayviviano@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      scripts=["mudpie_sample_generator.py"],
      license='LICENSE.md',
    )
