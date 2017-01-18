$head

namespace spyks {

template <typename value_type, typename time_type=double>
struct $name {
        static const size_t N_PARAM = $n_param;
        static const size_t N_STATE = $n_state;
        static const size_t N_FORCING = $n_forcing;
        typedef typename std::array<value_type, N_STATE> state_type;
        value_type const * $param_var;
        value_type const * $forcing_var;
        time_type dt;

        $name (value_type const * p, value_type const * f, time_type forcing_dt)
             : $param_var(p), $forcing_var(f), dt(forcing_dt) {}

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
integrate(Model & model, typename Model::state_type x, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        size_t nsteps = floor(tmax / dt);
        auto obs = pyarray_dense<Model>(nsteps);
        auto stepper = resetting_euler<state_type>();
        ode::integrate_const(stepper, model, x, 0.0, tmax, dt, obs);
        return obs.X;
}

}

using spyks::$name;

PYBIND11_PLUGIN($name) {
        typedef double value_type;
        typedef double time_type;
        typedef $name<value_type, time_type> model;
        py::module m("$name", "$descr");
        py::class_<model>(m, "model")
                .def("__init__",
                     [](model &m,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                        time_type forcing_dt) {
                             auto pptr = static_cast<value_type const *>(params.data());
                             auto dptr = static_cast<value_type const *>(forcing.data());
                             new (&m) model(pptr, dptr, forcing_dt);
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
                              model::state_type x0,
                              py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                              time_type forcing_dt, time_type stepping_dt) -> py::array {
                      auto pptr = static_cast<value_type const *>(params.data());
                      py::buffer_info forcing_info = forcing.request();
                      auto dptr = static_cast<value_type const *>(forcing_info.ptr);
                      time_type tmax = forcing_info.shape[0] * forcing_dt;
                      model model(pptr, dptr, forcing_dt);
                      return spyks::integrate(model, x0, tmax, stepping_dt);
              },
              "Integrates model from starting state x0 over the duration of the forcing timeseries",
              "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
        m.def("integrate", &spyks::integrate<model>);
        return m.ptr();
}
