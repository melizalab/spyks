# spyks

[![Build Status](https://travis-ci.org/melizalab/spyks.png?branch=master)](https://travis-ci.org/melizalab/spyks)

spyks is a tool for building and simulating simple, dynamical neuron models. It places a strong emphasis on efficient modification of parameters, because it's primarily intended for use in data assimilation applications where the goal is to estimate parameters and unmeasured states from recorded data. It is very much a work in progress.

In general, a dynamical neuron model comprises a set of ordinary differential equations `dX/dt = F(X, t, θ)` that determine how the state of the neuron `X` evolves in time. The components of the state vector include the membrane voltage and the (in)activation states of the various currents that contribute to the voltage. In the classical Hodgkin-Huxley model, these additional variables, called `m`, `h`, and `n`, obey first-order kinetics. In many phenomenological models, the voltage diverges when the neuron spikes, and the model must also include a rule for resetting the state when this occurs. The vector `θ` comprises parameters that govern the behavior of the system (for example, the reversal potential of the fast sodium current or the half-activation voltage of a delayed-rectifier potassium current). Parameters are assumed to be constant.

In spyks, models are specified in a YAML document that contains at a minimum, the equations of motion in symbolic form and values for all of the parameters and state variables. The model may also include a reset rule. Values are specified with physical units for dimensional analysis and reduced errors from unit mismatches.

### Notes

- To compile models, you'll need a C++ compiler that's compliant with the C++11 or later standard and the boost libraries installed (specifically boost odeint)
- `pybind11` may not install its headers correctly if it's installed when `python setup.py install` is called. If you get errors about missing headers running `spykscc`, try reinstalling `pip uninstall pybind11; pip install pybind11`
