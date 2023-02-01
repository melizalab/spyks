$head

namespace spyks {

/** A stepper for integrating a resetting neuron model using Euler's method */
template <typename state_type>
struct resetting_euler {
        typedef typename state_type::value_type value_type;
        typedef state_type deriv_type;
        typedef double time_type;
        typedef unsigned short order_type;
        typedef boost::numeric::odeint::stepper_tag stepper_category;

        static order_type order( void ) { return 1; }

        template <typename System>
        void do_step(System system, state_type & x, time_type t, time_type dt) {
                if (!system.reset(x)) {
                        deriv_type dxdt;
                        system(x, dxdt, t);
                        for (size_t i = 0; i < x.size(); ++i)
                                x[i] += dt * dxdt[i];
                        system.clip(x);
                }
        }
};

template <typename T, typename interpolator_type>
struct $name {
        static const size_t N_PARAM = $n_param;
        static const size_t N_STATE = $n_state;
        static const size_t N_FORCING = $n_forcing;
        typedef T value_type;
        typedef typename std::array<value_type, N_STATE> state_type;
        typedef typename interpolator_type::time_type time_type;
        value_type const * $param_var;
        interpolator_type $forcing_var;
	resetting_euler<state_type> stepper;

        $name (value_type const * p, interpolator_type f)
             : $param_var(p), $forcing_var(f) {}

        void operator()(state_type const & $state_var,
                        state_type & $deriv_var,
                        time_type $time_var) const {
                $forcing
                $substitutions
                $system
        }

        bool reset(state_type & X) const {
                bool rp = $reset_predicate ;
                if (rp) {
                        $reset_state
                }
                return rp;
        }

        void clip(state_type & X) const {
                $clip
        }
};

} // namespace spyks

PYBIND11_MODULE($name, m) {
        typedef double value_type;
        typedef double time_type;
        typedef spyks::nn_interpolator<value_type, time_type> interpolator;
        typedef spyks::$name<value_type, interpolator> model_type;
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
				auto _forcing = interpolator(forcing, forcing_dt);
				return new model_type(pptr, _forcing);
			}),
		     "Instantiate a new model with parameters and external forcing",
		     "params"_a, "forcing"_a, "forcing_dt"_a)
                .def("__call__",
		     [](model_type const & m, state_type const & X, time_type t) {
			     state_type out;
			     m(X, out, t);
			     return out;
		     },
		     "Compute the system function given state X at time t",
		     "state"_a, "time"_a)
                .def("reset",
		     [](model_type const & m, state_type & X) {
			     bool r = m.reset(X);
			     return std::make_pair(r, X);
		     },
		     "Reset the model if it meets the reset conditions. Returns (was_reset, new_state)",
		     "state"_a)
                .def("clip",
		     [](model_type const & m, state_type & X) {
			     m.clip(X);
			     return X;
		     },
		     "Clip the model state variables if they meet the clippin conditions. Returns new state."
		     "state"_a)
		.def("integrate",
		     [](model_type & model, state_type state, time_type dt) {
			     time_type tmax = model.forcing.get_max_time();
			     size_t nsteps = ceil(tmax / dt);
			     auto obs = spyks::pyarray_dense<model_type>(nsteps);
			     ode::integrate_const(model.stepper, model, state, 0.0, tmax, dt, obs);
			     return obs.X;
		     },
		     "Integrates model from state(t=0) through the duration of the forcing timeseries",
		     "state"_a, "stepping_dt"_a);

}


