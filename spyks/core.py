# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pint as pq
import numpy as nx

# need a global unit registry in order for quantities to be compatible
ureg = pq.UnitRegistry()

def parse_quantity(x):
    try:
        return ureg.parse_expression(x)
    except AttributeError:
        return x * ureg.dimensionless


def load_model(fname):
    """ Loads a model descriptor file. """
    import ruamel.yaml as yaml
    fp = open(fname, "r")
    model = yaml.load(fp, yaml.RoundTripLoader)  # preserves order
    # parse quantities and units
    model['parameters'] = [(param, parse_quantity(value)) for param, value in model['parameters'].items()]
    model['forcing'] = [(param, parse_quantity(value)) for param, value in model['forcing'].items()]
    model['state'] = [(param, parse_quantity(value)) for param, value in model['state'].items()]
    return model


def to_array(mapping):
    """ Converts named list of quantities into a dimensionless array. """
    return nx.fromiter((v.magnitude for k, v in mapping), dtype='d', count=len(mapping))


def to_mapping(values, mapping):
    """ Binds values in vector to names and units in mapping """
    return [(k, x * getattr(v, 'units', v)) for  (k, v), x in zip(mapping, values) ]


def update_values(model, **kwargs):
    """ Updates values in the model. Keyword arguments must match existing keys """
    for key, values in kwargs.items():
        model[key] = to_mapping(values, model[key])


math_funs = {'exp': nx.exp, 'tanh': nx.tanh, 'pow': nx.power}

def context(model):
    context = dict(model['state'], **math_funs)
    context.update(model['forcing'])
    context.update(model['parameters'])
    return context


def deriv(model):
    """ Evaluates the model system dX/dt = F(X, t) """
    ctx = context(model)
    return [(variable, eval(eqn, {}, ctx)) for variable, eqn in model['equations'].items()]


def check_reset(model):
    """ Evaluates the model's reset condition """
    return eval(model['reset']['predicate'], {}, context(model))


def reset(model):
    """ Returns the model's post-reset state """
    ctx = context(model)
    return [(variable, eval(eqn, {}, ctx)) for variable, eqn in model['reset']['state'].items()]
