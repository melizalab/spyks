#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from spyks import __version__
from setuptools import setup
import sys
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 4):
    raise RuntimeError("Python version 2.7 or >= 3.4 required.")


cls_txt = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

long_desc = """
spyks is a tool for building and simulating simple, dynamical neuron models. It
places a strong emphasis on efficient modification of parameters, because it's
primarily intended for use in data assimilation applications where the goal is
to estimate parameters and unmeasured states from recorded data. It is very much
a work in progress.
"""

setup(
    name="spyks",
    version=__version__,
    description="Generate fast c++ code to integrate neuron models",
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],

    packages=['spyks'],
    package_data={'spyks': ["templates/*.cpp"]},
    entry_points={'console_scripts': ['spykscc = spyks.build:compile_script']},
    install_requires=[
        "pybind11>=2.2.0",
        "Pint>=0.7",
        "ruamel.yaml>=0.13",
        "sympy>=1.0",
        "numpy>=1.11"
    ],

    author="Tyler Robbins",
    maintainer='C Daniel Meliza',
    url="https://github.com/melizalab/spyks",
    test_suite='nose.collector'
)
