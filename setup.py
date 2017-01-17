#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from spyks import __version__
import os
import glob
import logging
from spyks import build
from setuptools import setup, Extension

log = logging.getLogger('spyks')   # root logger
ch = logging.StreamHandler()
formatter = logging.Formatter("[%(name)s] %(message)s")
loglevel = logging.INFO
log.setLevel(loglevel)
ch.setLevel(loglevel)  # change
ch.setFormatter(formatter)
log.addHandler(ch)

def make_extension(modelfile):
    from spyks.core import load_model
    path = os.path.dirname(modelfile)
    model = load_model(modelfile)
    cppfile = os.path.join(path, model["name"] + ".cpp")
    if build.should_rebuild(modelfile, cppfile):
        build.write_cppfile(model, cppfile)
    else:
        log.info("%s: code is up to date", model["name"])
    return Extension("spyks.models.{}".format(model["name"]),
                    sources=[cppfile],
                    include_dirs=[build.get_pybind_include(), build.get_pybind_include(user=True),
                                  build.get_boost_include(), build.get_include()],
                    language="c++")


_modelfiles = ["models/nakl.yml", "models/biocm.yml"]
_models = [make_extension(fname) for fname in _modelfiles]

setup(
    name="spyks",
    version="0.5.0",
    packages=['spyks'],
    headers=[
        "include/spyks/integrators.h"
    ],

    cmdclass = {'build_ext': build.BuildExt},
    ext_modules= _models,

    entry_points={'console_scripts': ['spykscc = spyks.build:compile_script'] },

    description="minimalist spiking neuron model library",
    author="Tyler Robbins",
    author_email="tdr5wc at the domain 'virginia.edu'",
)
