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
    I = 500
    N = 1000
    dt = 0.05
    params = adex_params[0]['params']
    x0 = adex_params[0]['state']
    data = nx.ones(N) * I
    model = models.AdEx(params, models.timeseries(data, dt))
    X = models.integrate(model, x0, dt)
    # with these parameters, there should be exactly one spike at 555
    events = (X[:,0] > 29.9).nonzero()[0]
    assert_equal(events.size, 1)
    assert_equal(events[0], 555)

nakl_params = [
    {'params': [1., 120., 50., 20., -77., 0.3, -54.4,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30., 1.],
     'forcing': [0.],
     'state': [-70., 0., 0., 0.]},
    {'params': [1., 120., 50., 20., -77., 0.3, -54.4,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30., 1.],
     'forcing': [100.],
     'state': [-70., 0., 0.2, 0.1]},
    {'params': [1.02, 70.7, 55., .38, -85., 0.054, -65,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30., .042],
     'forcing': [0.],
     'state': [-65., 0.1, 0.2, 0.1]},
]

def py_nakl(X, params, Iinj):
    C, gna, Ena, gk, Ek, gl, El, vm, dvm, tm0, tm1, vmt, dvmt, vh, dvh, \
        th0, th1, vht, dvht, vn, dvn, tn0, tn1, vnt, dvnt, Isa = tuple(params)
    V, m, h, n = tuple(X)

    dV = 1/C * ((gna*m*m*m*h*(Ena - V)) +
                (gk*n*n*n*n*(Ek - V)) +
                (gl*(El-V)) + Iinj/Isa)

    taum = tm0 + tm1 * (1-pow(nx.tanh((V - vmt)/dvmt),2))
    m0 = (1+nx.tanh((V - vm)/dvm))/2
    dm = (m0 - m)/taum

    tauh = th0 + th1 * (1-pow(nx.tanh((V - vht)/dvht),2))
    h0 = (1+nx.tanh((V - vh)/dvh))/2
    dh = (h0 - h)/tauh

    taun = tn0 + tn1 * (1-pow(nx.tanh((V - vnt)/dvnt),2))
    n0 = (1+nx.tanh((V - vn)/dvn))/2
    dn = (n0 - n)/taun

    return (dV, dm, dh, dn)


def test_nakl_dxdt():
    def compare_nakl(params, forcing, state):
        inj = models.timeseries(forcing, 0.05)
        model = models.NaKL(params, inj)
        dXdt = model(state, 0)
        assert_true(nx.allclose(dXdt, py_nakl(state, params, forcing[0])))
    for tvals in nakl_params:
        yield compare_nakl, tvals['params'], tvals['forcing'], tvals['state']


def test_nakl_integration():
    I = 50
    N = 1000
    dt = 0.05
    params = nakl_params[0]['params']
    x0 = nakl_params[0]['state']
    data = nx.ones(N) * I
    model = models.NaKL(params, models.timeseries(data, dt))
    X = models.integrate(model, x0, dt)
    # with these parameters, there should be six spikes
