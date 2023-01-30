# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for building extension modules """
import logging
import os
import sys

import pybind11

log = logging.getLogger("spyks")  # root logger


def should_rebuild(modelfile, cppfile):
    t_model = os.path.getmtime(modelfile)
    t_cpp = os.path.getmtime(cppfile)
    return t_cpp < t_model


def write_cppfile(model, fname, simplify=True):
    """Renders model to C++ code and writes to fname"""
    from .codegen import render, simplify_equations

    if simplify:
        log.info("%s: simplifying equations", model["name"])
        model = simplify_equations(model)
    code = render(model)
    with open(fname, "wt") as fp:
        fp.write(code)
    log.info("%s: wrote %s", model["name"], fname)


# this is pretty hacky but trying to do it through setuptools was far too fragile
def build_module(cppfile, name, path, **kwargs):
    import subprocess
    import sysconfig

    outfile = os.path.join(path, name) + sysconfig.get_config_var("EXT_SUFFIX")
    command = [
        "c++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++11",
        "-fPIC",
    ]
    if sys.platform == "darwin":
        command += ["-undefined", "dynamic_lookup"]
    if kwargs.get("optimize", False):
        command += ["-fvisibility=hidden", "-ffast-math", "-flto"]
    command += [
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{pybind11.get_include()}",
        cppfile,
        "-o",
        outfile,
    ]
    log.info("%s: compiling extension module", name)
    log.debug(" command: %s", " ".join(command))
    subprocess.run(command)
    log.info("%s: complete - extension module in %s/", name, path)


def jit(model):
    """Compile a model file into an extension module and load it.

    Compilation can be quite slow for complex models, and this function doesn't
    do any kind of caching, so it's more for illustration purposes.

    """
    import spyks.validate as spkv

    spkv.check_symbols(model)
    spkv.check_equations(model)


def compile_script(argv=None):
    import argparse

    import spyks.validate as spkv
    from spyks import __version__
    from spyks.core import load_model

    p = argparse.ArgumentParser(
        description="compile a spyks model file into a python extension module"
    )
    p.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument("--skip-compile", help="skip compilation step", action="store_true")
    p.add_argument(
        "--skip-codegen", help="skip code generation step", action="store_true"
    )
    p.add_argument(
        "--optimize",
        help="enable aggressive compiler optimizations",
        action="store_true",
    )
    p.add_argument("model", help="the model descriptor file to compile")
    p.add_argument(
        "target",
        help="the path to put the module (default same as model file)",
        nargs="?",
    )
    args = p.parse_args(argv)

    ch = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    loglevel = logging.DEBUG if args.debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)  # change
    ch.setFormatter(formatter)
    log.addHandler(ch)

    try:
        model = load_model(args.model, load_base=False)
    except ValueError as e:
        log.info("error parsing model: %s", e)
        # log.info("%s extends a base model - compile the base model instead", args.model)
        return 0
    log.info("%s: validating model", model["name"])
    spkv.check_symbols(model)
    spkv.check_equations(model)
    path = args.target or os.path.dirname(args.model)
    cppfile = os.path.join(path, model["name"] + ".cpp")
    if not args.skip_codegen:
        write_cppfile(model, cppfile)

    if not args.skip_compile:
        build_module(
            cppfile,
            model["name"],
            path,
            version=model["version"],
            optimize=args.optimize,
        )
