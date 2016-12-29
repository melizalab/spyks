#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utility.hpp"

namespace spyks { namespace models {

struct adex {
        static const size_t N_PARAM = 10;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;
        double const * params;
        double const * forcing;
        double dt;

        adex(double const * parameters, double const * forcing, double forcing_dt)
                : params(parameters), forcing(forcing), dt(forcing_dt) {}

        void operator()(state_type const & X, state_type & dXdt, double t) const {
                double I = interpolate(t, forcing, dt, N_FORCING)[0];
                dXdt[0] = 1/params[0] * (-params[1] * (X[0] - params[2]) +
                                         params[1] * params[3] * std::exp((X[0]-params[4])/params[3])
                                         - X[1] + I);
                dXdt[1] = 1/params[5] * (params[6] * (X[0] - params[2]) - X[1]);
        }

        bool check_reset(state_type & X) const {
                if (X[0] < params[9])
                        return false;
                X[0] = params[9];
                return true;
        }

        void reset_state(state_type & X) const {
                X[0] = params[7];
                X[1] += params[8];
        }

};

}}

namespace py = pybind11;
using namespace pybind11::literals;
using spyks::models::adex;

PYBIND11_PLUGIN(adex) {
    py::module m("adex", "adaptive exponential model");
    py::class_<adex>(m, "model")
            .def("__init__", [](adex &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                py::array_t<double, py::array::c_style | py::array::forcecast> forcing,
                                double forcing_dt) {
                         auto pptr = static_cast<double const *>(params.data());
                         auto dptr = static_cast<double const *>(forcing.data());
                         new (&m) adex(pptr, dptr, forcing_dt);
                 })
            .def("__call__", [](adex const & m, adex::state_type const & X, double t) -> adex::state_type {
                            adex::state_type out;
                            m(X, out, t);
                            return out;
                    })
            .def("check_reset", [](adex const & m, adex::state_type & X) -> std::pair<bool, adex::state_type> {
                            bool r = m.check_reset(X);
                            return make_pair(r, X);
                    })
            .def("reset_state", [](adex const & m, adex::state_type & X) -> adex::state_type {
                            m.reset_state(X);
                            return X;
                    });
    return m.ptr();
}
