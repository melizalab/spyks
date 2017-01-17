# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Generates C++ extension module code for a model

Example:

model = core.load_model(modelfile)
template = pkgutil.get_data("spyks.templates", "model.cpp")
# symbolic simplification is time-intensive but may speed up C++ code
model = simplify_equations(model)
code = render(model, template)

"""
import operator
import logging
import sympy as sp
from spyks.core import n_params, n_state, n_forcing

value_type = "double"
param_var = "p"
state_var = "X"
deriv_var = "dXdt"
forcing_var = "forcing"
time_var = "t"

log = logging.getLogger('spyks')   # root logger

def simplify_equations(model):
    m = model.copy()
    m['equations'] = [(n, expr.simplify()) for n, expr in model['equations']]
    return m


def symbol_replacements(model):
    P = sp.IndexedBase(param_var, shape=(n_params(model),))
    X = sp.IndexedBase(state_var, shape=(n_state(model),))
    state_subs = {n: X[i] for i, (n, v) in enumerate(model["state"])}
    param_subs = {n: P[i] for i, (n, v) in enumerate(model["parameters"])}
    state_subs.update(param_subs)
    return state_subs


def fmt_subst(name, expr):
    return "const {} {} = {};".format(value_type, sp.ccode(name), sp.ccode(expr))


def fmt_dx(i, expr):
    return "{}[{}] = {};".format(deriv_var, i, sp.ccode(expr))


def fmt_forcing(i, s):
    templ = "const {} {} = interpolate({}, {}, dt, N_FORCING)[{}];"
    return templ.format(value_type, s, time_var, forcing_var, i)


def fmt_reset(n, expr):
    return "{} = {};".format(sp.ccode(n), sp.ccode(expr))


def fmt_clip(n, expr):
    return "if ({0} > {1}) {0} = {1};".format(sp.ccode(n), sp.ccode(expr))


def fmt_systemf(model):
    """Produces c code from expressions, applying cse"""
    repl = symbol_replacements(model)
    forcing_s = "\n".join(fmt_forcing(i, s)
                          for i, (s, v) in enumerate(model["forcing"]))
    log.info("%s: eliminating common subexpressions", model["name"])
    subs, exprs = sp.cse(expr for n, expr in model["equations"])
    subs_s = "\n".join(fmt_subst(n, expr.subs(repl)) for n, expr in subs)
    expr_s = "\n".join(fmt_dx(i, expr.subs(repl)) for i, expr in enumerate(exprs))
    return {
        "forcing": forcing_s,
        "substitutions": subs_s,
        "system": expr_s
    }


def fmt_resetf(model):
    if "reset" not in model: return {}
    repl = symbol_replacements(model)
    out = {
        "reset_predicate": sp.ccode(model['reset']['predicate'].subs(repl)),
        "reset_state": "\n".join(fmt_reset(n.subs(repl), expr.subs(repl)) for n, expr in model['reset']['state'])
    }
    try:
        out["clip"] = "\n".join(fmt_clip(n.subs(repl), expr.subs(repl)) for n, expr in model['reset']['clip'])
    except KeyError:
        pass
    return out


def get_template(model):
    import pkgutil
    if "reset" in model:
        name = "model_reset.cpp"
    else:
        name = "model_continuous.cpp"
    return pkgutil.get_data("spyks.templates", name).decode("utf-8")


def render(model):
    import string
    context = dict(name=model["name"],
                   descr=model["description"],
                   n_param=n_params(model),
                   n_state=n_state(model),
                   n_forcing=n_forcing(model),
                   param_var=param_var,
                   forcing_var=forcing_var,
                   state_var=state_var,
                   deriv_var=deriv_var,
                   time_var=time_var)
    context.update(fmt_systemf(model))
    context.update(fmt_resetf(model))

    templ = string.Template(get_template(model))
    return templ.substitute(context)
