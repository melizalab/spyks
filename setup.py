#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from spyks import __version__
import os
from setuptools import setup

VERSION = '0.6.1'
cls_txt = """
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Topic :: Internet :: WWW/HTTP
Topic :: Internet :: WWW/HTTP :: Dynamic Content
"""

setup(
    name="spyks",
    version=VERSION,
    description="Generate fast c++ code to integrate neuron models",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    classifiers=[x for x in cls_txt.split("\n") if x],

    packages=['spyks'],
    package_data={'spyks': ["templates/*.cpp"]},
    entry_points={'console_scripts': ['spykscc = spyks.build:compile_script'] },
    install_requires = [
        "pybind11>=2.1.1",
        "Pint>=0.7",
        "ruamel.yaml>=0.13",
        "sympy>=1.0"
    ],

    author="Tyler Robbins",
    maintainer='C Daniel Meliza',
)
