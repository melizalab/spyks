# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Tools for manipulating and fitting kinetic functions"""
import numpy as np

from spyks import core


def ode_fun(model, var, fname="inf"):
    """Create a lambda to evaluate var_∞(V) or var_τ(V) or var_α(V) or var_β(V)"""
    from sympy import Symbol, lambdify

    ode = core.kinetic_ode(var, **model["eqns_unparsed"][var])
    context = {Symbol(n): v.magnitude for n, v in model["parameters"]}
    return lambdify([Symbol("V")], getattr(ode, fname).subs(context))


def tanh_inf(p, V):
    vm, dvm = p[:2]
    return (1 + np.tanh((V - vm) / dvm)) / 2


def tanh_tau(p, V):
    tm0, tm1, vmt, dvmt = p[:4]
    return tm0 + tm1 * (1 - np.tanh((V - vmt) / dvmt) ** 2)


def tanh_tau2(p, V):
    tm0, tm1, vmt, dvmt1, tm2, dvmt2 = p[:6]
    return tm0 + (
        1 + tm1 * np.tanh((V - vmt) / dvmt1) * (1 - tm2 * np.tanh((V - vmt) / dvmt2))
    )


def tanh_tau3(p, V):
    tm0, tm1, vmt, dvmt, deltt = p[:6]
    return (
        tm0
        + deltt
        + 0.5
        * (1 - np.tanh(V - vmt))
        * tm1
        * (1 - np.tanh((V - vmt) / dvmt) ** 2 - deltt)
    )


def exp_inf(p, V):
    nam_v, nam_dv, pow = p[:3]
    return np.power((1 + np.exp(-(V - nam_v) / nam_dv)), pow)


def exp_tau(p, V):
    nam_t0, nam_t1, nam_tv, nam_tdv1, nam_t2, nam_tdv2 = p[:6]
    return nam_t0 + 1 / (
        nam_t1 * np.exp((V - nam_tv) / nam_tdv1)
        + nam_t2 * np.exp(-(V - nam_tv) / nam_tdv2)
    )


def refit_kinetics(V, modelfun, candidatefun, p0):
    """Refit the kinetic function modelfun to candidatefun.

    For example, if the kinetics are specified using some funky \alpha(V) and
    \beta(V) formulation, you can estimate the best parameters for an x_\inf(V)
    and \tau(V) formulation based on hyperbolic tangents.

    V - the voltage values at which to evaluate the function
    modelfun - a callable f(V) that evaluates the target function over V
    candidatefun - a callable f(p, V) that evaluates the candidate function over V with params p
    p0 - initial parameter guess
    """
    from scipy import optimize

    return optimize.leastsq(
        lambda p, x, y: candidatefun(p, x) - y, p0[:], args=(V, modelfun(V))
    )
