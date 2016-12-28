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


def param_values(model):
    """ Extracts model parameter values into a dimensionless array. """
    p = model['parameters']
    return nx.fromiter((v.magnitude for v in p.values()), dtype='d', count=len(p))


def units(x):
    """ Returns the units of x. If x is a Unit object, returns itself """
    return getattr(x, 'units', x)


def bind(values, mapping):
    """ Binds values in vector to names and units in mapping """
    return [(k, x * units(v)) for  (k, v), x in zip(mapping, values) ]

math_funs = {'exp': nx.exp, 'tanh': nx.tanh, 'pow': nx.power}

def deriv(model):
    """ Evaluates the model system dX/dt = F(X, t) """
    context = dict(model['state'], **math_funs)
    context.update(model['forcing'])
    context.update(model['parameters'])
    return [(variable, eval(eqn, {}, context)) for variable, eqn in model['equations'].items()]
