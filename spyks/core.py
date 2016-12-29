# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pint as pq

# need a global unit registry in order for quantities to be compatible
ureg = pq.UnitRegistry()

def parse_quantity(x):
    if isinstance(x, (float, int)):
        return x * ureg.dimensionless
    else:
        return ureg.parse_expression(x)


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
    from numpy import fromiter
    return fromiter((v.magnitude for k, v in mapping), dtype='d', count=len(mapping))


def to_mapping(values, mapping):
    """ Binds values in vector to names and units in mapping """
    return [(k, x * getattr(v, 'units', v)) for  (k, v), x in zip(mapping, values) ]


def update_model(model, **kwargs):
    """ Updates values in the model. Keyword arguments must match existing keys """
    for key, values in kwargs.items():
        model[key] = to_mapping(values, model[key])
