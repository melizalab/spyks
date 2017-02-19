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
        value_type const * $param_var;
        interpolator_type $forcing_var;

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

template<typename Model>
py::array
integrate(Model & model, py::array_t<typename Model::value_type> x0, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        state_type x;
        std::copy_n(x0.data(), Model::N_STATE, x.begin());
        size_t nsteps = floor(tmax / dt);
        auto obs = pyarray_dense<Model>(nsteps);
        auto stepper = resetting_euler<state_type>();
        ode::integrate_const(stepper, model, x, 0.0, tmax, dt, obs);
        return obs.X;
}

}

PYBIND11_PLUGIN($name) {
        typedef double value_type;
        typedef double time_type;
        typedef spyks::nn_interpolator<value_type, time_type> interpolator;
        typedef spyks::$name<value_type, interpolator> model;
        py::module m("$name", "$descr");
        py::class_<model>(m, "model")
                .def("__init__",
                     [](model &m,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                        time_type forcing_dt) {
                             auto pptr = static_cast<value_type const *>(params.data());
                             auto _forcing = interpolator(forcing, forcing_dt);
                             new (&m) model(pptr, _forcing);
                     })
                .def("__call__", [](model const & m, model::state_type const & X, time_type t) {
                                model::state_type out;
                                m(X, out, t);
                                return out;
                        })
                .def("reset", [](model const & m, model::state_type & X) {
                                bool r = m.reset(X);
                                return std::make_pair(r, X);
                        })
                .def("clip", [](model const & m, model::state_type & X) {
                                m.clip(X);
                                return X;
                        });


        m.def("integrate", [](py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                              py::array_t<value_type, py::array::c_style | py::array::forcecast> x0,
                              py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                              time_type forcing_dt, time_type stepping_dt) -> py::array {
                      auto pptr = static_cast<value_type const *>(params.data());
                      time_type tmax = forcing.shape(0) * forcing_dt;
                      auto _forcing = interpolator(forcing, forcing_dt);
                      model model(pptr, _forcing);
                      return spyks::integrate(model, x0, tmax, stepping_dt);
              },
              "Integrates model from starting state x0 over the duration of the forcing timeseries",
              "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
        m.def("integrate", &spyks::integrate<model>);
        return m.ptr();
}
