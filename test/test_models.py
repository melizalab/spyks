# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as nx

from spyks import core, models


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


def test_adex_dxdt():
    pymodel = core.load_model("models/adex.yml")
    def compare_adex(params, forcing, state):
        inj = models.timeseries(forcing, 0.05)
        model = models.AdEx(params, inj)
        dXdt = model(state, 0)
        core.update_values(pymodel, state=state, forcing=forcing, parameters=params)
        pydXdt = core.to_array(core.deriv(pymodel))
        assert_true(nx.allclose(dXdt, pydXdt))
    for tvals in adex_params:
        yield compare_adex, tvals['params'], tvals['forcing'], tvals['state']


def test_adex_reset():
    pymodel = core.load_model("models/adex.yml")
    def compare_reset(params, forcing, state):
        model = models.AdEx(params, models.timeseries(forcing, 0.05))
        reset, pre_state = model.check_reset(state)
        core.update_values(pymodel, state=state, forcing=forcing, parameters=params)
        pyreset = core.check_reset(pymodel)
        assert_equal(reset, pyreset)
        if reset:
            post_state = model.reset_state(state)
            pypost_state = core.to_array(core.reset(pymodel))
            assert_true(nx.allclose(post_state, pypost_state))
        # else:
        #     assert_true(nx.allclose(state, pypost_state))
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

def test_nakl_dxdt():
    pymodel = core.load_model("models/nakl.yml")
    def compare_nakl(params, forcing, state):
        inj = models.timeseries(forcing, 0.05)
        model = models.NaKL(params, inj)
        dXdt = model(state, 0)
        core.update_values(pymodel, state=state, forcing=forcing, parameters=params)
        pydXdt = core.to_array(core.deriv(pymodel))
        assert_true(nx.allclose(dXdt, pydXdt))
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


def test_biocm():
    pymodel = core.load_model("models/biocm.yml")
    params = core.to_array(pymodel['parameters'])
    forcing = core.to_array(pymodel['forcing'])
    state = core.to_array(pymodel['state']).tolist()
    inj = models.timeseries(forcing, 0.05)
    model = models.biocm(params, inj)
    dXdt = model(state, 0)
    pydXdt = core.to_array(core.deriv(pymodel))
    assert_true(nx.allclose(dXdt, pydXdt))

def test_biocm_integration(I=50):
    N = 1000
    dt = 0.05
    pymodel = core.load_model("models/biocm.yml")
    params = core.to_array(pymodel['parameters'])
    x0 = core.to_array(pymodel['state']).tolist()
    data = nx.ones(N) * I
    model = models.biocm(params, models.timeseries(data, dt))
    X = models.integrate(model, x0, dt)
    return X
