# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as nx

from spyks import models


adex_params = [
    {'params': [250, 30, -70.6, 2.0, -55, 144, 4, -70.6, 80.5, 30, 1],
     'forcing': [0],
     'state': [-70, 0]},
    {'params': (200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30, 1),
     'forcing': [0.05],
     'state': [-50, 0]},
    {'params': nx.asarray([200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30, 1]),
     'forcing': [0.05],
     'state': [50, 10]},
]


def py_adex(X, params, Iinj):
    C, gl, el, delt, vt, tw, a, vr, b, h, R = list(params)
    dX = [1 / C *  (-gl * (X[0] - el) + gl * delt * nx.exp((X[0] - vt) / delt) - X[1] + R * Iinj),
          1 / tw * (a * (X[0] - el) - X[1])]
    return dX


def test_adex_dxdt():
    def compare_adex(model, params, forcing, state):
        model.set_params(params)
        model.set_forcing(forcing, 0.05)
        dXdt = model(state, 0)
        assert_true(nx.allclose(dXdt, py_adex(state, params, forcing[0])))
    model = models.AdEx()
    for tvals in adex_params:
        yield compare_adex, model, tvals['params'], tvals['forcing'], tvals['state']

def test_adex_reset():
    def compare_reset(model, params, state):
        model.set_params(params)
        newval, reset = model.reset(state)
        if state[0] < params[9]:
            assert_equal(reset, False)
            assert_equal(newval[0], state[0])
            assert_almost_equal(newval[1], state[1])
        else:
            assert_equal(reset, True)
            assert_almost_equal(newval[0], params[7])
            assert_almost_equal(newval[1], state[1] + params[8])
    model = models.AdEx()
    for tvals in adex_params:
        yield compare_reset, model, tvals['params'], tvals['state']
