#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

from setuptools import setup, find_packages, Extension

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

try:
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

compiler_settings = {
    "include_dirs" : [numpy_include]
}

# if sys.platform == "win32" :
#     include_dirs = ["C:/Boost/include/boost-1_32","."]
#     libraries=["boost_python-mgw"]
#     library_dirs=['C:/Boost/lib']
# elif sys.platform == 'darwin':
#     include_dirs = ["/opt/local/include", numpy_include]
#     libraries=["boost_python-mt"]
#     library_dirs=["/opt/local/lib"]
# else:
#     include_dirs = ["/usr/include/boost",numpy_include]
#     libraries=["boost_python"]
#     library_dirs=['/usr/local/lib']

_models = Extension("spyks.models", sources=["spyks/models" + SUFFIX,
                                             "src/neurons.cpp"],
                    language="c++",
                    **compiler_settings)

setup(
    name="spyks",
    version="0.2.0",
    packages= find_packages(exclude=["*test*"]),
    cmdclass = {'build_ext': build_ext},
    ext_modules=[_models],

      description="minimalist spiking neuron model library",
      author="Tyler Robbins",
      author_email="tdr5wc at the domain 'virginia.edu'",


)
