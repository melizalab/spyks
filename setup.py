#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02070000:
    raise RuntimeError("Python 2.7 or higher required")

import os,glob
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import setuptools

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_boost_include(object):
    def __str__(self):
        if sys.platform == "win32" :
            return "C:/Boost/include/boost-1_32"
        elif sys.platform == 'darwin':
            return "/opt/local/include"
        else:
            return "/usr/include/boost"


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)

_modelfiles = glob.glob("models/*.cpp")
_models = [Extension("spyks.models.{}".format(os.path.splitext(os.path.basename(fname))[0]),
                    sources=[fname],
                    include_dirs=[get_pybind_include(), get_pybind_include(user=True),
                                  get_boost_include(), "src/"],
                    language="c++")
           for fname in _modelfiles]

setup(
    name="spyks",
    version="0.2.0",
    packages= find_packages(exclude=["*test*"]),
    cmdclass = {'build_ext': BuildExt},
    ext_modules= _models,

    description="minimalist spiking neuron model library",
    author="Tyler Robbins",
    author_email="tdr5wc at the domain 'virginia.edu'",
)
