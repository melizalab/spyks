# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Functions to validate models, primarily with respect to dimensional correctness.
"""

def context(model):
    """ Returns the model's components and related math functions in a dict for eval'ing equations """
    import numpy as nx
    context = {'exp': nx.exp, 'tanh': nx.tanh, 'pow': nx.power}
    context.update(model['state'])
    context.update(model['forcing'])
    context.update(model['parameters'])
    return context


def deriv(model):
    """ Evaluates the model system dX/dt = F(X, t). Will throw error if dimensional analysis fails. """
    ctx = context(model)
    return [(variable, eval(eqn, {}, ctx)) for variable, eqn in model['equations'].items()]


def reset(model):
    """ Returns the model's post-reset state, or False if it did not reset """
    ctx = context(model)
    if eval(model['reset']['predicate'], {}, context(model)):
        return [(variable, eval(eqn, {}, ctx)) for variable, eqn in model['reset']['state'].items()]
    else:
        return False


def check_symbols(model):
    from itertools import chain
    from .core import symbols
    lhs, rhs = symbols(model)
    vars = {n for n,v in chain(model["state"], model["parameters"], model["forcing"])}
    d = vars.symmetric_difference(str(s) for s in rhs)
    if len(d) > 0:
        raise ValueError("unmatched variables in model: {}".format(d))


def check_equations(model):
    from .core import ureg
    dt = 1 * ureg.ms
    X = model['state']
    dX = deriv(model)           # will raise error if dimensions are bad
    for ((n1, v1), (n2, v2)) in zip(X, dX):
        if n1 != n2: raise ValueError("Name mismatch between state ({}) and derivative ({})".format(n1, n2))
        try:
            x = v1 + dt * v2
        except:
            raise ValueError("Units mismatch between state ({}) and derivative ({}) for {}".format(v1.units,
                                                                                                   v2.units,
                                                                                                   n1))

def check_forcing(model, forcing):
    if forcing.ndim > 2:
        raise ValueError("Forcing must be 1-D or 2-D")
    N = len(model['forcing'])
    D = forcing.shape[1] if forcing.ndim > 1 else 1
    if N != D:
        raise ValueError("Forcing component count ({}) does not match model ({})".format(D, N))
