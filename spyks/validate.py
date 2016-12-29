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


def check_reset(model):
    """ Evaluates the model's reset condition """
    return eval(model['reset']['predicate'], {}, context(model))


def reset(model):
    """ Returns the model's post-reset state """
    ctx = context(model)
    return [(variable, eval(eqn, {}, ctx)) for variable, eqn in model['reset']['state'].items()]
