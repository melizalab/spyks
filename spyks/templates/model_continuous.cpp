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
};

template<typename Model>
py::array
integrate(Model & model, py::array_t<typename Model::value_type> x0, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        state_type x;
        std::copy_n(x0.data(), Model::N_STATE, x.begin());
        size_t nsteps = ceil(tmax / dt);
        auto obs = pyarray_dense<Model>(nsteps);
        // auto stepper = ode::runge_kutta4<state_type>();
        // ode::integrate_const(stepper, std::ref(model), x, 0.0, tmax, dt, obs);
        auto stepper = ode::runge_kutta_dopri5<state_type>();
        ode::integrate_const(ode::make_dense_output(1.0e-5, 1.0e-5, stepper),
                             std::ref(model), x, 0.0, tmax, dt, obs);
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
