#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from spyks import __version__
import os
import glob
from setuptools import setup

setup(
    name="spyks",
    version="0.6.0",
    packages=['spyks'],
    package_data={'spyks': ["templates/*.cpp"]},

    entry_points={'console_scripts': ['spykscc = spyks.build:compile_script'] },

    description="minimalist spiking neuron model library",
    author="Tyler Robbins",
    author_email="tdr5wc at the domain 'virginia.edu'",
)
