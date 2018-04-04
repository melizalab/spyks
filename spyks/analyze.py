# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interpreting and analyzing model output"""
# python 3 compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

from spyks import core


def state_fun(model, state):
    """Create a lambda that will evaluate functions of the model state"""
    from spyks.codegen import state_replacements
    from spyks.validate import evalf
    repl = state_replacements(model)
    units = [v.units for n, v in model["state"]]
    context = {"X": [state[:, i] * u for i, u in enumerate(units)]}
    context.update(model["parameters"])
    # context = {n: v.magnitude for n, v in model['parameters']}
    # context["X"] = state.T
    return lambda expr: evalf(expr.subs(repl), context)


def currents(model, state):
    """Calculate I_x(t) for a completed model"""
    f = state_fun(model, state)
    return [f(expr).to("pA") for expr in core.currents(model)]


def conductances(model, state):
    """Calculate g_x(t) for a completed model"""
    f = state_fun(model, state)
    return [f(expr).to("nS") for expr in core.conductances(model)]
