$head

namespace spyks {

template <typename T, typename interpolator_type>
struct $name {
        static const size_t N_PARAM = $n_param;
        static const size_t N_STATE = $n_state;
        static const size_t N_FORCING = $n_forcing;
        typedef T value_type;
        typedef typename std::array<value_type, N_STATE> state_type;
        typedef typename interpolator_type::time_type time_type;
	// parameters are stored in a C array to keep access fast
        value_type const * $param_var;
        interpolator_type $forcing_var;
	ode::runge_kutta_dopri5<state_type> stepper;

        $name (value_type const * p, interpolator_type f)
             : $param_var(p), $forcing_var(f) {}

        void operator()(state_type const & $state_var,
                        state_type & $deriv_var,
                        time_type $time_var) const {
                $forcing
                $substitutions
                $system
        }
};

} // namespace spyks

PYBIND11_MODULE($name, m) {
        typedef double value_type;
        typedef double time_type;
        typedef spyks::nn_interpolator<value_type, time_type> interpolator_type;
        typedef spyks::$name<value_type, interpolator_type> model_type;
	typedef model_type::state_type state_type;
        m.doc() = "$descr";
        m.attr("name") = py::cast("$name");
        m.attr("__version__") = py::cast($version);
        py::class_<model_type>(m, "model")
                .def(py::init([](py::array_t<value_type> params,
                                 py::array_t<value_type> forcing,
                                 time_type forcing_dt)
			{
                             auto pptr = static_cast<value_type const *>(params.data());
                             auto _forcing = interpolator_type(forcing, forcing_dt);
                             return new model_type(pptr, _forcing);
			}),
		     "Instantiate a new model_type with parameters and external forcing",
		     "params"_a, "forcing"_a, "forcing_dt"_a)
                .def("__call__",
		     [](model_type const & m, state_type const & X, time_type t) {
			     state_type out;
			     m(X, out, t);
			     return out;
		     },
		     "Compute the system function given state X at time t",
		     "state"_a, "time"_a)
		.def("update_forcing",
		     [](model_type & model, py::array_t<value_type> forcing, time_type forcing_dt) {
                             model.forcing = interpolator_type(forcing, forcing_dt);
		     },
		     "Update the forcing for the model")
		.def("step",
		     [](model_type & model, state_type state, time_type t, time_type dt) {
			     model.stepper.do_step(std::ref(model), state, t, dt);
			     return state;
		     },
		     "Iterate the model_type one step from state x at time t. The stepper may retain information "
		     "about the derivative between calls, so the model should be reinitialized for a totally new input.",
		     "state"_a, "time"_a, "step"_a)
		.def("integrate",
		     [](model_type & model, state_type state, time_type dt) {
			     time_type tmax = model.forcing.get_max_time();
			     size_t nsteps = ceil(tmax / dt);
			     auto obs = spyks::pyarray_dense<model_type>(nsteps);
			     auto dense_stepper = ode::make_dense_output(1.0e-5, 1.0e-5, model.stepper);
			     ode::integrate_const(dense_stepper, std::ref(model), state, 0.0, tmax, dt, obs);
			     return obs.X;
		     },
		     "Integrates model from state(t=0) through the duration of the forcing timeseries",
		     "state"_a, "step"_a);
}
