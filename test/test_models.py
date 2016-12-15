# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as nx

from spyks import models


adex_params = [
    {'params': [250, 30, -70.6, 2.0, -55, 144, 4, -70.6, 80.5, 30, 1],
     'forcing': [0.],
     'state': [-70., 0]},
    {'params': (200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30, 1),
     'forcing': [0.05],
     'state': [-50., 0]},
    {'params': nx.asarray([200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30, 1]),
     'forcing': [0.05],
     'state': [50., 10]},
]


def py_adex(X, params, Iinj):
    C, gl, el, delt, vt, tw, a, vr, b, h, R = list(params)
    dX = [1 / C *  (-gl * (X[0] - el) + gl * delt * nx.exp((X[0] - vt) / delt) - X[1] + R * Iinj),
          1 / tw * (a * (X[0] - el) - X[1])]
    return dX


def test_univariate_timeseries():
    N = 1000
    dt = 0.05
    data = nx.random.randn(N)
    ts = models.timeseries(data, dt)
    assert_equal(ts.dt(), dt)
    assert_equal(ts.duration(), N * dt)
    assert_equal(ts.dimension(), 1)
    assert_equal(data[0], ts(0, 0))
    assert_equal(data[-1], ts(0, ts.duration() - ts.dt()))


def test_multivariate_timeseries():
    N = 1000
    D = 10
    dt = 0.05
    data = nx.random.randn(N, D)
    ts = models.timeseries(data, dt)
    assert_equal(ts.dt(), dt)
    assert_equal(ts.duration(), N * dt)
    assert_equal(ts.dimension(), D)
    assert_equal(data[0,0], ts(0, 0))
    assert_equal(data[-1,D-1], ts(D-1, ts.duration() - ts.dt()))


def test_adex_dxdt():
    def compare_adex(params, forcing, state):
        inj = models.timeseries(forcing, 0.05)
        model = models.AdEx(params, inj)
        dXdt = model(state, 0)
        assert_true(nx.allclose(dXdt, py_adex(state, params, forcing[0])))
    for tvals in adex_params:
        yield compare_adex, tvals['params'], tvals['forcing'], tvals['state']


def test_adex_reset():
    def compare_reset(params, forcing, state):
        model = models.AdEx(params, models.timeseries(forcing, 0.05))
        reset, pre_state = model.check_reset(state)
        if state[0] < params[9]:
            assert_equal(reset, False)
            assert_equal(pre_state[0], state[0])
            assert_equal(pre_state[1], state[1])
        else:
            assert_equal(reset, True)
            assert_almost_equal(pre_state[0], params[9])
            post_state = model.reset_state(state)
            assert_almost_equal(post_state[0], params[7])
            assert_almost_equal(post_state[1], state[1] + params[8])
    for tvals in adex_params:
        yield compare_reset, tvals['params'], tvals['forcing'], tvals['state']


def test_adex_integration():
    N = 1000
    dt = 0.05
    params = adex_params[0]['params']
    x0 = adex_params[0]['state']
    data = nx.random.randn(N)
    model = models.AdEx(params, models.timeseries(data, dt))
    X, t = models.integrate_adex(model, x0, dt)
    assert_true(((nx.diff(t) - dt) < 1e-10).all())
