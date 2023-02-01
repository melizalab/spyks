# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Core functions for model descriptors

In spyks, models are specified by a map with following required keys: name
(string), state (mapping), equations (mapping), and parameters (mapping). Optional
keys include forcing (mapping) and reset (with predicate and state).

The convention in this package is that "mapping" is a list of (name, value)
tuples. Values must be specified with physical units (e.g., '5 mV') to permit
dimensional analysis; if units are not specified the quantity is assumed to be
dimensionless. An array, on the other hand, does not have names or quantities.

"""
import os

import pint as pq
import sympy as sp

# need a global unit registry in order for quantities to be compatible
ureg = pq.UnitRegistry()


def n_params(model):
    """Number of parameters in model"""
    return len(model["parameters"])


def n_state(model):
    """Number of state variables in model"""
    return len(model["equations"])


def n_forcing(model):
    """Number of forcing terms in model"""
    return len(model["forcing"])


def param_names(model):
    """Names of parameters in model"""
    return tuple(n for n, v in model["parameters"])


def state_names(model):
    """Names of state variables in model"""
    return tuple(n for n, v in model["state"])


def forcing_names(model):
    """Names of forcing terms in model"""
    return tuple(n for n, v in model["forcing"])


def symbols(model):
    lh = set()
    rh = set()
    for n, eq in model["equations"]:
        lh.add(sp.Symbol(n))
        rh.update(eq.free_symbols)
    try:
        reset = model["reset"]
        rh.update(reset["predicate"].free_symbols)
        for n, eq in reset["state"]:
            rh.update(eq.free_symbols)
        for n, eq in reset.get("clip", []):
            rh.update(eq.free_symbols)
    except KeyError:
        pass
    return lh, rh


def to_array(mapping):
    """Converts named vector of quantities into a dimensionless array."""
    from numpy import fromiter

    return fromiter((v.magnitude for k, v in mapping), dtype="d", count=len(mapping))


def to_scaled_array(mapping):
    """Converts named vector of quantities into a dimensionless array."""
    from numpy import fromiter

    return fromiter(
        (v.to_base_units().magnitude for k, v in mapping), dtype="d", count=len(mapping)
    )


def to_mapping(array, mapping):
    """Binds values in array to names and units in mapping. Size of array must match mapping."""
    assert len(array) == len(mapping)
    return [(k, x * getattr(v, "units", v)) for (k, v), x in zip(mapping, array)]


def parse_quantity(x):
    if isinstance(x, (float, int)):
        return x * ureg.dimensionless
    else:
        return ureg.parse_expression(x)


class kinetic_ode(object):
    """Represents an ODE for a 1st order kinetic process

    Initialize with alpha & beta or inf and tau expressions
    """

    _alpha = None
    _beta = None
    _inf = None
    _tau = None

    def __init__(self, n, **eqns):
        self._var = sp.Symbol(n)
        if "alpha" in eqns:
            self._alpha = sp.sympify(eqns["alpha"])
        if "beta" in eqns:
            self._beta = sp.sympify(eqns["beta"])
        if "inf" in eqns:
            self._inf = sp.sympify(eqns["inf"])
        if "tau" in eqns:
            self._tau = sp.sympify(eqns["tau"])
        if not (
            (self._alpha is not None and self._beta is not None)
            or (self._inf is not None and self._tau is not None)
        ):
            raise ValueError("must specify alpha/beta or inf/tau")

    @property
    def expr(self):
        if self._alpha is not None and self._beta is not None:
            return self._alpha * (1 - self._var) - self._beta * self._var
        else:
            return (self._inf - self._var) / self._tau

    @property
    def inf(self):
        return self._inf or self._alpha / (self.alpha + self.beta)

    @property
    def tau(self):
        return self._tau or 1 / (self.alpha + self.beta)

    @property
    def alpha(self):
        return self._alpha or (self.inf / self.tau)

    @property
    def beta(self):
        return self._beta or ((1 - self.inf) / self.tau)


def parse_equation(n, s):
    """Parse equations of motion to sympy expression.

    This function tries to be smart about different formulations of state
    variables ODEs. It can understand the alpha/beta specification and the
    inf/tau specification. Otherwise it will throw an error. Errors are also
    thrown on badly formed equations.

    """
    import ruamel.yaml as yaml

    try:
        if isinstance(s, (dict, yaml.comments.CommentedMap)):
            expr = kinetic_ode(n, **s).expr
        else:
            expr = sp.sympify(s)
        return (n, expr)
    except (TypeError, sp.SympifyError):
        raise ValueError("unable to parse equation spec for {}".format(n))


def parse(model):
    """Parse equations, variables, and parameters of model into sympy objects"""
    from copy import copy
    from itertools import starmap

    model["eqns_unparsed"] = copy(model["equations"])
    model["equations"] = list(starmap(parse_equation, model["equations"].items()))
    try:
        reset = model["reset"]
        reset["predicate"] = sp.sympify(reset["predicate"])
        reset["state"] = [
            (sp.Symbol(n), sp.sympify(s)) for n, s in reset["state"].items()
        ]
        reset["clip"] = [
            (sp.Symbol(n), sp.sympify(s)) for n, s in reset["clip"].items()
        ]
    except KeyError:
        pass
    # parse quantities and units
    for k in ("parameters", "forcing", "state"):
        if k in model:
            model[k] = [
                (param, parse_quantity(value)) for param, value in model[k].items()
            ]
    return model


def _mapping_updater(new):
    """Returns an f that updates mapping elements with values in new, converting units as needed"""

    def f(kv):
        n, v = kv
        if n in new:
            return (n, parse_quantity(new[n]).to(v.units))
        else:
            return (n, v)

    return f


def load_model(doc, load_base=True):
    """Loads a model descriptor from a file, string, or stream

    If the model descriptor has a "base" field, by default this function will
    use the supplied model to update the base model's default parameter and
    state values. If load_base is not True, then the function will raise a
    ValueError.

    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    if os.path.exists(doc):
        fname = doc
        fp = open(doc, "r")
        model = yaml.load(fp)
        fp.close()
    else:
        fname = ""
        model = yaml.load(doc)
    if "base" in model:
        if not load_base:
            raise ValueError("model extends %s but load_base is False" % model["base"])
        basefile = os.path.join(os.path.dirname(fname), model["base"] + ".yml")
        base = load_model(basefile)  # will get parsed
        if "parameters" in model:
            base["parameters"] = list(
                map(_mapping_updater(dict(model["parameters"])), base["parameters"])
            )
        if "state" in model:
            base["state"] = list(
                map(_mapping_updater(dict(model["state"])), base["state"])
            )
        return base
    else:
        return parse(model)


def update_model(model, **kwargs):
    """Updates model with arrays of values.

    Keyword arguments (e.g., parameters, forcing, state) must match existing
    keys, and array size must match size of model vector.

    """
    for key, array in kwargs.items():
        model[key] = to_mapping(array, model[key])


def get_param_value(model, name):
    """Return parameter value from model"""
    return dict(model["parameters"]).get(name)


def set_param_value(model, name, value):
    """Update parameter value in the model"""
    # this is somewhat clunky because the parameters are in a list, not a dict
    for i, (param, old) in enumerate(model["parameters"]):
        if param == name:
            model["parameters"][i] = (param, value)
            break


def load_module(model, path=None):
    """Loads the extension module associated with model.

    If `path` of extension module file is supplied, temporarily adds it to
    sys.path. Raises an error if the name or version don't match.

    """
    import importlib
    import sys

    if path is not None:
        sys.path.append(path)
    try:
        mdl = importlib.import_module(model["name"])
        if mdl.name != model["name"]:
            raise ImportError(
                "extension module name ({}) doesn't match model name ({})".format(
                    mdl.name, model["name"]
                )
            )
        if mdl.__version__ != model["version"]:
            raise ImportError(
                "extension module version ({}) doesn't match descriptor ({})".format(
                    mdl.__version__, model["version"]
                )
            )
    finally:
        if path is not None:
            sys.path.remove(path)
    return mdl


def make_model(model, forcing, forcing_dt, path=None):
    """Instantiates the extension module class for model with specified forcing"""
    module = load_module(model, path)
    params = to_array(model["parameters"])
    return module.model(params, forcing, forcing_dt)


def currents(model):
    """Extract intrinsic current terms from the model

    This function assumes that the model is biophysical: the model's first
    equation is for voltage, specified by current conservation.

    """
    dV = model["equations"][0][1]
    I_net = sp.fraction(dV)[0]
    forcing_vars = forcing_names(model)
    return (term for term in I_net.args if str(term) not in forcing_vars)


def conductances(model):
    """Extract intrinsic conductance terms from the model

    This function works by dividing out any arguments from currents that depend on the voltage
    """
    V = sp.Symbol("V")
    return (
        sp.prod(t for t in term.args if V not in t.free_symbols)
        for term in currents(model)
    )


def conductance_names(model):
    """Extract names of conductances in the model"""
    return (str(expr.args[0]) for expr in currents(model))
