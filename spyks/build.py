# -*- coding: utf-8 -*-
# -*- mode: python -*-

import os
import sys
import sysconfig
import pybind11
import distutils
import logging
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

log = logging.getLogger('spyks')   # root logger


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        return pybind11.get_include(self.user)


def get_boost_include():
    if sys.platform == "win32" :
        return "C:/Boost/include/boost-1_32"
    elif sys.platform == 'darwin':
        return "/opt/local/include"
    else:
        return "/usr/include/boost"


def get_include(*args, **kwargs):
    try:
        from pip import locations
        return os.path.dirname(
            locations.distutils_scheme('spyks', *args, **kwargs)['headers'])
    except ImportError:
        return 'include'


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


class ImportCppExt(Extension):
    """ A custom Extension class for specifying installation destination """
    def __init__(self, libdest, *args, **kwargs):
        self.libdest = libdest
        Extension.__init__(self, *args, **kwargs)


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
            if has_flag(self.compiler, '-ffast-math'):
                opts.append('-ffast-math')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


class CustomDestBuildExt(BuildExt):
    """A custom build extension for specifying target path"""
    def copy_extensions_to_source(self):
        for ext in self.extensions:
            fullname = self.get_ext_fullname(ext.name)
            filename = self.get_ext_filename(fullname)
            src_filename = os.path.join(self.build_lib, filename)
            dest_filename = os.path.join(ext.libdest, os.path.basename(filename))

            distutils.file_util.copy_file(
                src_filename, dest_filename,
                verbose = self.verbose, dry_run = self.dry_run
            )


def should_rebuild(modelfile, cppfile):
    t_model = os.path.getmtime(modelfile)
    t_cpp   = os.path.getmtime(cppfile)
    return t_cpp < t_model


def write_cppfile(model, fname, simplify=True):
    """ Renders model to C++ code and writes to fname"""
    from .codegen import render, simplify_equations
    if simplify:
        log.info("%s: simplifying equations", model["name"])
        model = simplify_equations(model)
    code = render(model)
    with open(fname, "wt") as fp:
        fp.write(code)
    log.info("%s: wrote %s", model["name"], fname)


def build_module(cppfile, name, path, **kwargs):
    import shutil, tempfile
    ext = ImportCppExt(
        path,
        name,
        language = "c++",
        sources = [cppfile],
        include_dirs = [get_pybind_include(), get_pybind_include(True),
                        get_boost_include(), get_include()],
    )

    build_path = tempfile.mkdtemp()
    args = ['build_ext', '--inplace', '-q']
    args.append('--build-temp=' + build_path)
    args.append('--build-lib=' + build_path)
    log.info("%s: compiling extension module", name)
    setup(name = name,
          version = kwargs.get("version", "0.0.1-SNAPSHOT"),
          ext_modules = [ext],
          script_args = args,
          cmdclass = { "build_ext": CustomDestBuildExt })
    shutil.rmtree(build_path)
    log.info("%s: complete - extension module in %s", name, path)


# def import_model(modelfile):
#     from .core import load_model

#     model = load_model(modelfile)
#     log.info("%s: validating model", model["name"])
#     spkv.check_symbols(model)
#     spkv.check_equations(model)
#     path = args.target or os.path.dirname(args.model)
#     cppfile = os.path.join(path, model["name"] + ".cpp")
#     write_cppfile(model, cppfile)

#     path = os.path.dirname(modelfile)
#     cppfile = os.path.join(path, model["name"] +


def compile_script(argv=None):
    from .core import load_model
    import spyks.validate as spkv
    import argparse

    p = argparse.ArgumentParser(description="compile a spyks model file into a python extension module")
    p.add_argument("model", help="the model descriptor file to compile")
    p.add_argument("target", help="the path to put the module (default same as model file)", nargs='?')
    p.add_argument("--skip-compile", help="skip compilation step", action="store_true")
    p.add_argument("--skip-codegen", help="skip code generation step", action="store_true")
    args = p.parse_args(argv)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    loglevel = logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)  # change
    ch.setFormatter(formatter)
    log.addHandler(ch)

    model = load_model(args.model)
    log.info("%s: validating model", model["name"])
    spkv.check_symbols(model)
    spkv.check_equations(model)
    path = args.target or os.path.dirname(args.model)
    cppfile = os.path.join(path, model["name"] + ".cpp")
    if not args.skip_codegen:
        write_cppfile(model, cppfile)

    if not args.skip_compile:
        build_module(cppfile, model["name"], path, version=model["version"])
