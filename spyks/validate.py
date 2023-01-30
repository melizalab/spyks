# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Functions to validate models, primarily with respect to dimensional correctness.
"""


def context(model):
    """Returns the model's components and related math functions in a dict for eval'ing equations"""
    import math

    import numpy as nx

    context = {
        "math": math,
        "exp": nx.exp,
        "tanh": nx.tanh,
        "pow": nx.power,
        "sqrt": nx.sqrt,
        "expm1": nx.expm1,
    }
    context.update(model["state"])
    context.update(model["forcing"])
    context.update(model["parameters"])
    return context


def evalf(eqn, context):
    from sympy.printing.lambdarepr import lambdarepr

    return eval(lambdarepr(eqn), {}, context)


def _deriv_g(model):
    """Generator that evaluates each component of the model system dX/dt = F(X, t). Will throw error if dimensional analysis fails."""


def deriv(model):
    """Evaluates the model system dX/dt = F(X, t). Will throw error if dimensional analysis fails."""
    ctx = context(model)
    return [(variable, evalf(eqn, ctx)) for variable, eqn in model["equations"]]


def reset(model):
    """Returns the model's post-reset state, or False if it did not reset"""
    ctx = context(model)
    if evalf(model["reset"]["predicate"], ctx):
        return [
            (variable, evalf(eqn, ctx)) for variable, eqn in model["reset"]["state"]
        ]
    else:
        return False


def check_symbols(model):
    """Raises ValueError iff mismatch between equations and parameter/state/forcing vectors"""
    from itertools import chain

    from .core import symbols

    lhs, rhs = symbols(model)
    vars = {n for n, v in chain(model["state"], model["parameters"], model["forcing"])}
    d = vars.symmetric_difference(str(s) for s in rhs)
    if len(d) > 0:
        raise ValueError("unmatched variables in model: {}".format(d))


def check_equations(model):
    """Raises ValueError iff mismatch between names or units of equations and state vector"""
    from pint.errors import DimensionalityError

    from .core import ureg

    ctx = context(model)
    dt = 1 * ureg.ms
    X = model["state"]
    dX = list()
    # check dimensions
    for variable, eqn in model["equations"]:
        try:
            d = evalf(eqn, ctx)
            dX.append((variable, d))
        except DimensionalityError as e:
            raise ValueError(
                "Dimensionality error in d{}/dt = {}: {}".format(variable, eqn, e)
            )
    for ((n1, v1), (n2, v2)) in zip(X, dX):
        if n1 != n2:
            raise ValueError(
                "Name mismatch between state ({}) and derivative ({})".format(n1, n2)
            )
        try:
            _ = v1 + dt * v2
        except:
            raise ValueError(
                "Units mismatch between ({}) and derivative ({}) for {}".format(
                    v1.units, v2.units, n1
                )
            )


def check_forcing(model, forcing):
    """Raises ValueError iff mismatch between number of components in forcing array and model"""
    if forcing.ndim > 2:
        raise ValueError("Forcing must be 1-D or 2-D")
    N = len(model["forcing"])
    D = forcing.shape[1] if forcing.ndim > 1 else 1
    if N != D:
        raise ValueError(
            "Forcing component count ({}) does not match model ({})".format(D, N)
        )
