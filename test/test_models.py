# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import assert_almost_equal, assert_equal, assert_true
import numpy as nx

import spyks as s
import spyks.validate as sv

from models import adex, nakl, biocm

adex_params = [
    {'params': [250, 30, -70.6, 2.0, -55, 144, 4, -70.6, 80.5, 30],
     'forcing': [0.],
     'state': [-70., 0]},
    {'params': (200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30),
     'forcing': [0.05],
     'state': [-50., 0]},
    {'params': nx.asarray([200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30]),
     'forcing': [0.05],
     'state': [50., 10]},
]


def test_adex_dxdt():
    pymodel = s.load_model("models/adex.yml")
    def compare_adex(params, forcing, state):
        # inj = models.timeseries(forcing, 0.05)
        model = adex.model(params, forcing, 0.05)
        dXdt = model(state, 0)
        s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
        pydXdt = s.to_array(sv.deriv(pymodel))
        assert_true(nx.allclose(dXdt, pydXdt))
    for tvals in adex_params:
        yield compare_adex, tvals['params'], tvals['forcing'], tvals['state']


def test_adex_reset():
    pymodel = s.load_model("models/adex.yml")
    def compare_reset(params, forcing, state):
        model = adex.model(params, forcing, 0.05)
        reset, new_state = model.reset(state)
        s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
        pypost_state = sv.reset(pymodel)
        if reset:
            pypost_state = s.to_array(pypost_state)
            assert_true(nx.allclose(new_state, pypost_state))
        else:
            assert_equal(reset, pypost_state)
            assert_true(nx.allclose(state, new_state))
    for tvals in adex_params:
        yield compare_reset, tvals['params'], tvals['forcing'], tvals['state']


def test_adex_integration():
    I = 500
    N = 1000
    dt = 0.05
    params = adex_params[0]['params']
    x0 = adex_params[0]['state']
    data = nx.ones(N) * I
    model = adex.model(params, data, dt)
    X = adex.integrate(params, x0, data, dt, dt)
    X2 = adex.integrate(model, x0, data.size * dt, dt)
    assert_true(nx.allclose(X, X2))
    # with these parameters, there should be exactly one spike at 555
    events = (X[:,0] > 29.9).nonzero()[0]
    assert_equal(events.size, 1)
    assert_equal(events[0], 555)
    return X


nakl_params = [
    {'params': [1., 120., 50., 20., -77., 0.3, -54.4,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30.],
     'forcing': [0.],
     'state': [-70., 0., 0., 0.]},
    {'params': [1., 120., 50., 20., -77., 0.3, -54.4,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30.],
     'forcing': [100.],
     'state': [-70., 0., 0.2, 0.1]},
    {'params': [1.02, 70.7, 55., .38, -85., 0.054, -65,
                -40., 15., 0.1, 0.4, -40., 15., -60., -15.,
                1., 7., -60., -15., -55., 30., 1., 5., -55., -30.],
     'forcing': [0.],
     'state': [-65., 0.1, 0.2, 0.1]},
]

def test_nakl_dxdt():
    pymodel = s.load_model("models/nakl.yml")
    def compare_nakl(params, forcing, state):
        model = nakl.model(params, forcing, 0.05)
        dXdt = model(state, 0)
        s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
        pydXdt = s.to_array(sv.deriv(pymodel))
        assert_true(nx.allclose(dXdt, pydXdt))
    for tvals in nakl_params:
        yield compare_nakl, tvals['params'], tvals['forcing'], tvals['state']


def test_nakl_integration(I=50):
    N = 10000
    dt = 0.05
    params = nakl_params[0]['params']
    x0 = nakl_params[0]['state']
    data = nx.ones(N) * I
    model = nakl.model(params, data, dt)
    X = nakl.integrate(params, x0, data, dt, dt)
    X2 = nakl.integrate(model, x0, data.size * dt, dt)
    assert_true(nx.allclose(X, X2))
    return X


def test_biocm():
    pymodel = s.load_model("models/biocm.yml")
    params = s.to_array(pymodel['parameters'])
    forcing = s.to_array(pymodel['forcing'])
    state = s.to_array(pymodel['state'])
    model = biocm.model(params, forcing, 0.05)
    dXdt = model(state.tolist(), 0)
    pydXdt = s.to_array(sv.deriv(pymodel))
    assert_true(nx.allclose(dXdt, pydXdt))


def test_biocm_integration(I=20):
    N = 5000
    dt = 0.05
    pymodel = s.load_model("models/biocm.yml")
    params = s.to_array(pymodel['parameters'])
    x0 = s.to_array(pymodel['state'])
    data = nx.ones(N) * I
    model = biocm.model(params, data, dt)
    X = biocm.integrate(params, x0, data, dt, dt)
    X2 = biocm.integrate(model, x0, data.size * dt, dt)
    assert_true(nx.allclose(X, X2))
    return X
