#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from spyks import __version__
from spyks.build import build
import os
import glob
from setuptools import setup

_modelfiles = glob.glob("models/*.yml")
# _models = [Extension("spyks.models.{}".format(os.path.splitext(os.path.basename(fname))[0]),
#                     sources=[fname],
#                     include_dirs=[get_pybind_include(), get_pybind_include(user=True),
#                                   get_boost_include(), "src/"],
#                     language="c++")
#            for fname in _modelfiles]

setup(
    name="spyks",
    version="0.5.0",
    packages=['spyks'],
    headers=[
        "include/spyks/integrators.h"
    ],

    # cmdclass = {'build_ext': BuildExt},
    # ext_modules= _models,

    entry_points={'console_scripts': ['spykscc = spyks.build:compile_script'] },

    description="minimalist spiking neuron model library",
    author="Tyler Robbins",
    author_email="tdr5wc at the domain 'virginia.edu'",
)
