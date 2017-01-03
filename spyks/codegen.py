# -*- coding: utf-8 -*-
# -*- mode: python -*-

import operator
import sympy as sp
from spyks.core import n_params, n_state, n_forcing

value_type = "double"
param_var = "p"
state_var = "X"
deriv_var = "dXdt"
forcing_var = "forcing"
time_var = "t"


def parse_equations(model):
    """Returns copy of model with parsed sympy expressions for model equations"""
    m = model.copy()
    m['equations'] = [(n, sp.sympify(s))
                      for n, s in model['equations'].items()]
    return m


def simplify_equations(model):
    m = model.copy()
    m['equations'] = [(n, expr.simplify()) for n, expr in model['equations']]
    return m


def print_subst(name, expr):
    return "const {} {} = {};".format(value_type, sp.ccode(name), sp.ccode(expr))


def print_dx(i, exp):
    return "{}[{}] = {};".format(deriv_var, i, sp.ccode(exp))


def print_forcing(i, s):
    templ = "const {} {} = interpolate({}, {}, dt, N_FORCING)[{}];"
    return templ.format(value_type, s, time_var, forcing_var, i)


def symbol_replacements(model):
    P = sp.IndexedBase(param_var, shape=(n_params(model),))
    X = sp.IndexedBase(state_var, shape=(n_state(model),))
    state_subs = {n: X[i] for i, (n, v) in enumerate(model["state"])}
    param_subs = {n: P[i] for i, (n, v) in enumerate(model["parameters"])}
    state_subs.update(param_subs)
    return state_subs


def to_ccode(model):
    """Produces c code from expressions, applying cse"""
    repl = symbol_replacements(model)
    subs, exprs = sp.cse(expr for n, expr in model["equations"])
    subs_s = "\n".join(print_subst(n, expr.subs(repl)) for n, expr in subs)
    expr_s = "\n".join(print_dx(i, expr.subs(repl))
                       for i, expr in enumerate(exprs))
    return "{}\n\n{}".format(subs_s, expr_s)


def replace_symbols(model):
    d = symbol_replacements(model)
    m = model.copy()
    m['equations'] = [(n, expr.subs(d)) for n, expr in model["equations"]]
    return m


def render(model, template):
    import string
    forcing_s = "\n".join(print_forcing(i, s)
                          for i, (s, v) in enumerate(model["forcing"]))
    code_s = to_ccode(model)
    fp = open(template, "r")
    templ = string.Template(open(template, "r").read())
    context = dict(name=model["name"],
                   descr=model["description"],
                   n_param=n_params(model),
                   n_state=n_state(model),
                   n_forcing=n_forcing(model),
                   param_var=param_var,
                   forcing_var=forcing_var,
                   state_var=state_var,
                   deriv_var=deriv_var,
                   time_var=time_var,
                   forcing=forcing_s,
                   code=code_s)
    return templ.substitute(context)
