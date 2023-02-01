# -*- coding: utf-8 -*-
# -*- mode: python -*-

import unittest
import numpy as np

import spyks.core as s
import spyks.validate as sv

passive_params = [
    {"params": [57, 5.5, -70.0, 0.0, -65.0], "forcing": [[0.0, 0.0]], "state": [-70.0]},
    {"params": (57, 5.5, -70.0, 0.0, -65.0), "forcing": [[1.0, 0.0]], "state": [-75.0]},
    {
        "params": np.asarray([57, 5.5, -70.0, 0.0, -65.0]),
        "forcing": [[0.0, 2.0]],
        "state": [-50.0],
    },
]


class TestPassiveModel(unittest.TestCase):
    _model_file = "models/passive_ei.yml"

    def test_passive_ei_deriv(self):
        pymodel = s.load_model(self._model_file)
        passive_ei = s.load_module(pymodel, "models")

        def compare(params, forcing, state):
            model = passive_ei.model(params, forcing, 0.05)
            dXdt = model(state, 0)
            s.update_model(pymodel, state=state, forcing=forcing[0], parameters=params)
            pydXdt = s.to_array(sv.deriv(pymodel))
            self.assertTrue(np.allclose(dXdt, pydXdt))

        for tvals in passive_params:
            yield compare, tvals["params"], tvals["forcing"], tvals["state"]

    def test_passive_ei_integration(self):
        pymodel = s.load_model(self._model_file)
        g_ex = 1.0
        g_inh = 0.5
        N = 2000
        dt = 0.05
        forcing = np.column_stack([np.ones(N) * g_ex, np.ones(N) * g_inh])
        model = s.make_model(pymodel, forcing, dt, "models")
        state = s.to_array(pymodel["state"])
        X = model.integrate(state, dt)
        # steady-state V should be a weighted average of the reversal potentials
        g_l = s.get_param_value(pymodel, "g_l")
        Evals = [s.get_param_value(pymodel, n) for n in ("E_l", "E_ex", "E_inh")]
        gvals = [g_l, g_ex * g_l.units, g_inh * g_l.units]
        V_steady = sum(g * e for e, g in zip(Evals, gvals)) / sum(gvals)
        events = (X[:, 0] > 29.9).nonzero()[0]
        self.assertAlmostEqual(V_steady.magnitude, X[-1, 0], places=2)
        self.assertEqual(len(events), 0)

    def test_passive_ei_step(self):
        pymodel = s.load_model(self._model_file)
        dt = 0.05
        forcing = [[0.0, 0.0]]
        model = s.make_model(pymodel, forcing, dt)
        state_0 = [-70.0]
        state_1 = model.step(state_0, 0, dt)
        # model should remain at equilibrium
        self.assertAlmostEqual(state_0[0], state_1[0])
        # add some excitation to the forcing
        model.update_forcing([[1.0, 0.0]], dt)
        state_2 = model.step(state_1, 0, dt)
        self.assertGreater(state_2[0], state_1[0])

    @unittest.skip(reason="bounds checking not yet implemented")
    def test_wrong_forcing_shape(self):
        pymodel = s.load_model(self._model_file)
        dt = 0.05
        forcing = [[0.0]]
        with self.assertRaises(Exception):
            model = s.make_model(pymodel, forcing, dt)


adex_params = [
    {
        "params": [250, 30, -70.6, 2.0, -55, 144, 4, -70.6, 80.5, 30],
        "forcing": [0.0],
        "state": [-70.0, 0],
    },
    {
        "params": (200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30),
        "forcing": [0.05],
        "state": [-50.0, 0],
    },
    {
        "params": np.asarray([200, 32, -65, 2.0, -56, 120, 5, -70.6, 80.5, 30]),
        "forcing": [0.05],
        "state": [50.0, 10],
    },
]


class TestAdExModel(unittest.TestCase):
    def test_adex_dxdt(self):
        pymodel = s.load_model("models/adex.yml")
        adex = s.load_module(pymodel, "models")

        def compare_adex(params, forcing, state):
            # inj = models.timeseries(forcing, 0.05)
            model = adex.model(params, forcing, 0.05)
            dXdt = model(state, 0)
            s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
            pydXdt = s.to_array(sv.deriv(pymodel))
            self.assertTrue(np.allclose(dXdt, pydXdt))

        for tvals in adex_params:
            yield compare_adex, tvals["params"], tvals["forcing"], tvals["state"]

    def test_adex_reset(self):
        pymodel = s.load_model("models/adex.yml")
        adex = s.load_module(pymodel, "models")

        def compare_reset(params, forcing, state):
            model = adex.model(params, forcing, 0.05)
            reset, new_state = model.reset(state)
            s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
            pypost_state = sv.reset(pymodel)
            if reset:
                pypost_state = s.to_array(pypost_state)
                self.assertTrue(np.allclose(new_state, pypost_state))
            else:
                self.assertEqual(reset, pypost_state)
                self.assertTrue(np.allclose(state, new_state))

        for tvals in adex_params:
            yield compare_reset, tvals["params"], tvals["forcing"], tvals["state"]

    def test_adex_integration(self):
        pymodel = s.load_model("models/adex.yml")
        Iinj = 500
        N = 1000
        dt = 0.05
        x0 = adex_params[0]["state"]
        forcing = np.ones(N) * Iinj
        model = s.make_model(pymodel, forcing, dt, "models")
        X = model.integrate(x0, dt)
        # with these parameters, there should be exactly one spike at 555
        events = (X[:, 0] > 29.9).nonzero()[0]
        self.assertEqual(events.size, 1)
        self.assertEqual(events[0], 555)


nakl_params = [
    {
        "params": [
            1.0,
            120.0,
            50.0,
            20.0,
            -77.0,
            0.3,
            -54.4,
            -40.0,
            15.0,
            0.1,
            0.4,
            -40.0,
            15.0,
            -60.0,
            -15.0,
            1.0,
            7.0,
            -60.0,
            -15.0,
            -55.0,
            30.0,
            1.0,
            5.0,
            -55.0,
            -30.0,
        ],
        "forcing": [0.0],
        "state": [-70.0, 0.0, 0.0, 0.0],
    },
    {
        "params": [
            1.0,
            120.0,
            50.0,
            20.0,
            -77.0,
            0.3,
            -54.4,
            -40.0,
            15.0,
            0.1,
            0.4,
            -40.0,
            15.0,
            -60.0,
            -15.0,
            1.0,
            7.0,
            -60.0,
            -15.0,
            -55.0,
            30.0,
            1.0,
            5.0,
            -55.0,
            -30.0,
        ],
        "forcing": [100.0],
        "state": [-70.0, 0.0, 0.2, 0.1],
    },
    {
        "params": [
            1.02,
            70.7,
            55.0,
            0.38,
            -85.0,
            0.054,
            -65,
            -40.0,
            15.0,
            0.1,
            0.4,
            -40.0,
            15.0,
            -60.0,
            -15.0,
            1.0,
            7.0,
            -60.0,
            -15.0,
            -55.0,
            30.0,
            1.0,
            5.0,
            -55.0,
            -30.0,
        ],
        "forcing": [0.0],
        "state": [-65.0, 0.1, 0.2, 0.1],
    },
]


class TestNaklModel(unittest.TestCase):
    def test_nakl_dxdt(self):
        pymodel = s.load_model("models/nakl.yml")
        nakl = s.load_module(pymodel, "models")

        def compare_nakl(params, forcing, state):
            model = nakl.model(params, forcing, 0.05)
            dXdt = model(state, 0)
            s.update_model(pymodel, state=state, forcing=forcing, parameters=params)
            pydXdt = s.to_array(sv.deriv(pymodel))
            self.assertTrue(np.allclose(dXdt, pydXdt))

        for tvals in nakl_params:
            yield compare_nakl, tvals["params"], tvals["forcing"], tvals["state"]

    def test_nakl_integration(self, Iinj=50):
        pymodel = s.load_model("models/nakl.yml")
        N = 10000
        dt = 0.05
        forcing = np.ones(N) * Iinj
        model = s.make_model(pymodel, forcing, dt, "models")
        x0 = nakl_params[0]["state"]
        X = model.integrate(x0, dt)
        self.assertEqual(X.shape, (N, len(x0)))


class TestBiocmModel(unittest.TestCase):
    def test_biocm(self):
        pymodel = s.load_model("models/biocm.yml")
        biocm = s.load_module(pymodel, "models")
        params = s.to_array(pymodel["parameters"])
        forcing = s.to_array(pymodel["forcing"])
        state = s.to_array(pymodel["state"])
        model = biocm.model(params, forcing, 0.05)
        dXdt = model(state.tolist(), 0)
        pydXdt = s.to_array(sv.deriv(pymodel))
        self.assertTrue(np.allclose(dXdt, pydXdt))

    def test_biocm_integration(self, Iinj=20):
        pymodel = s.load_model("models/biocm.yml")
        N = 5000
        dt = 0.05
        x0 = s.to_array(pymodel["state"])
        forcing = np.ones(N) * Iinj
        model = s.make_model(pymodel, forcing, dt)
        X = model.integrate(x0, dt)
        self.assertEqual(X.shape, (N, len(x0)))

    def test_biocm_step(self, Iinj=20):
        pymodel = s.load_model("models/biocm.yml")
        N = 5000
        dt = 0.05
        forcing = [0]
        model = s.make_model(pymodel, forcing, dt)
        state_0 = s.to_array(pymodel["state"])
        state_1 = model.step(state_0, 0.0, dt)
        # not exactly at equilibrium
        self.assertAlmostEqual(state_0[0], state_1[0], places=2)
        model.update_forcing([Iinj], dt)
        state_2 = model.step(state_1, 0.0, dt)
        self.assertGreater(state_2[0], state_1[0])
