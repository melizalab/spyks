# spyks

spyks is a tool for building and simulating simple, dynamical neuron models. It places a strong emphasis on efficient modification of parameters, because it's primarily intended for use in data assimilation applications where the goal is to estimate parameters and unmeasured states from recorded data. It is very much a work in progress.

In general, a dynamical neuron model comprises a set of ordinary differential equations $dX/dt = F(X, t, \theta)$ that determine how the state of the neuron $X$ evolves in time. The components of the state vector include the membrane voltage and the (in)activation states of the various currents that contribute to the voltage. In the classical Hodgkin-Huxley model, these additional variables are $m$, $h$, and $n$. In many phenomenological models, the voltage diverges when the neuron spikes, and the model must also include a rule for resetting the state when this occurs. The vector $\theta$ comprises parameters that govern the behavior of the system (for example, the reversal potential of the fast sodium current or the half-activation voltage of a delayed-rectifier potassium current). Parameters are assumed to be constant.

In spyks, models are specified in a YAML document that contains at a minimum, the equations of motion in symbolic form and values for all of the parameters and state variables. The model may also include a reset rule. Values are specified with physical units for dimensional analysis and reduced errors from unit mismatches.

spyks is very much a work in progress.
