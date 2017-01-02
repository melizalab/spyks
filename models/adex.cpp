#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utility.hpp"
#include "integrators.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace spyks { namespace models {

template <typename value_type, typename time_type=double>
struct adex {
        static const size_t N_PARAM = 10;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef typename std::array<value_type, N_STATE> state_type;
        value_type const * params;
        value_type const * forcing;
        time_type dt;

        adex(value_type const * parameters, value_type const * forcing, time_type forcing_dt)
                : params(parameters), forcing(forcing), dt(forcing_dt) {}

        void operator()(state_type const & X, state_type & dXdt, time_type t) const {
                value_type I = interpolate(t, forcing, dt, N_FORCING)[0];
                dXdt[0] = 1/params[0] * (-params[1] * (X[0] - params[2]) +
                                         params[1] * params[3] * std::exp((X[0]-params[4])/params[3])
                                         - X[1] + I);
                dXdt[1] = 1/params[5] * (params[6] * (X[0] - params[2]) - X[1]);
        }

        bool reset(state_type & X) const {
                if (X[0] < params[9])
                        return false;
                else {
                        X[0] = params[7];
                        X[1] += params[8];
                        return true;
                }
        }
};

}}

using spyks::models::adex;


PYBIND11_PLUGIN(adex) {
        typedef double value_type;
        typedef double time_type;
        typedef adex<value_type, time_type> dadex;
        py::module m("adex", "adaptive exponential model");
        py::class_<dadex >(m, "model")
            .def("__init__",
                 [](dadex &m,
                    py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                    py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                    time_type forcing_dt) {
                         auto pptr = static_cast<value_type const *>(params.data());
                         auto dptr = static_cast<value_type const *>(forcing.data());
                         new (&m) dadex(pptr, dptr, forcing_dt);
                 })
            .def("__call__", [](dadex const & m, dadex::state_type const & X, time_type t) {
                            dadex::state_type out;
                            m(X, out, t);
                            return out;
                    })
            .def("reset", [](dadex const & m, dadex::state_type & X) {
                            bool r = m.reset(X);
                            return std::make_pair(r, X);
                    });

    m.def("integrate", [](py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                          dadex::state_type x0,
                          py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                          time_type forcing_dt, time_type stepping_dt) -> py::array {
                  auto pptr = static_cast<value_type const *>(params.data());
                  py::buffer_info forcing_info = forcing.request();
                  auto dptr = static_cast<value_type const *>(forcing_info.ptr);
                  time_type tmax = forcing_info.shape[0] * forcing_dt;
                  dadex model(pptr, dptr, forcing_dt);
                  return spyks::integrate_reset(model, x0, tmax, stepping_dt);
          },
          "Integrates adex model from starting state x0 over the duration of the forcing timeseries",
          "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
    m.def("integrate", &spyks::integrate_reset<dadex>);
    return m.ptr();
}
