# spyks

[![Build Status](https://travis-ci.org/melizalab/spyks.png?branch=master)](https://travis-ci.org/melizalab/spyks)
[![DOI](https://zenodo.org/badge/102205384.svg)](https://zenodo.org/badge/latestdoi/102205384)

spyks is a tool for building and simulating simple, dynamical neuron models. It places a strong emphasis on efficient modification of parameters, because it's primarily intended for use in data assimilation applications where the goal is to estimate parameters and unmeasured states from recorded data. It is very much a work in progress.

In general, a dynamical neuron model comprises a set of ordinary differential equations `dX/dt = F(X, t, θ)` that determine how the state of the neuron `X` evolves in time. The components of the state vector include the membrane voltage and the (in)activation states of the various currents that contribute to the voltage. In the classical Hodgkin-Huxley model, these additional variables, called `m`, `h`, and `n`, obey first-order kinetics. In many phenomenological models, the voltage diverges when the neuron spikes, and the model must also include a rule for resetting the state when this occurs. The vector `θ` comprises parameters that govern the behavior of the system (for example, the reversal potential of the fast sodium current or the half-activation voltage of a delayed-rectifier potassium current). Parameters are assumed to be constant over the analysis interval.

In spyks, models are specified in a YAML document that contains at a minimum, the equations of motion in symbolic form and values for all of the parameters and state variables. The model may also include a reset rule. Values are specified with physical units for dimensional analysis and reduced errors from unit mismatches. Here's an example for the classic Hodkin-Huxley model. We're using hyperbolic tangents to parameterize the steady-state activation and time-constant functions for historical reasons (the derivatives are more numerically stable).

``` YAML
---
name: nakl
description: biophysical neuron model with minimal Na, K, leak conductances
author: dmeliza
version: 1.0
state:
  V: -70 mV
  m: 0
  h: 0
  n: 0
forcing:
  Iinj: 0 pA
equations:
  V: 1/C * ((gna*m*m*m*h*(Ena - V)) + (gk*n*n*n*n*(Ek - V)) + (gl*(El-V)) + Iinj)
  m:
    inf: (1+tanh((V - vm)/dvm))/2
    tau: (tm0 + tm1 * (1 - tanh((V - vmt)/dvmt)**2))
  h:
    inf: (1+tanh((V - vh)/dvh))/2
    tau: (th0 + th1 * (1 - tanh((V - vht)/dvht)**2))
  n:
    inf: (1+tanh((V - vn)/dvn))/2
    tau: (tn0 + tn1 * (1 - tanh((V - vnt)/dvnt)**2))
parameters:
  C: 250 pF
  gna: 120 nS
  Ena: 50 mV
  gk: 20 nS
  Ek: -77 mV
  gl: 0.3 nS
  El: -54.4 mV
  vm: -40 mV
  dvm: 15 mV
  tm0: 0.1 ms
  tm1: 0.4 ms
  vmt: -40 mV
  dvmt: 15 mV
  vh: -60 mV
  dvh: -15 mV
  th0: 1 ms
  th1: 7 ms
  vht: -60 mV
  dvht: -15 mV
  vn: -55 mV
  dvn: 40 mV
  tn0: 1 ms
  tn1: 5 ms
  vnt: -55 mV
  dvnt: -30 mV
```

Other examples are in the `models` directory. To compile this model into a Python extension module, just run `spykscc models/nakl.yml` in the shell.

Important: although spyks will do basic dimensional analysis to ensure that your parameters, equations, and state variables all have compatible dimensions, it currently cannot make sure that your parameters are scaled correctly. Capacitances have to be in `pF`, voltages in `mV`, conductances in `nS`, and times in `ms`. So for example, although you ought to be able to put in `0.25 nF` in the model descriptor above, you'll wind up with a 0.25 pF cell. Working on it.

In Python:

``` Python
import numpy as np
import spyks.core as spk

pymodel = spk.load_model("models/nakl.yml")
nakl = spk.load_module(pymodel, "models")
params = spk.to_array(pymodel['parameters'])
initial_state = spk.to_array(pymodel['state'])

Iinj = np.zeros(10000)
Iinj[2000:] = 50
dt = 0.05

X = nakl.integrate(params, initial_state, Iinj, dt, dt)

```

### Installation Notes

- To compile models, you'll need a C++ compiler that's compliant with the C++11 or later standard and the boost libraries installed (specifically boost odeint)
- If installing from source, use `pip install .` (or `pip install -e .` for a live/development install) rather than `python setup.py install`. `pybind11` may not install its headers correctly otherwise. If you get errors about missing headers running `spykscc`, try reinstalling `pip uninstall pybind11; pip install pybind11`
