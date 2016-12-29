#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utility.hpp"
#include "integrators.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
namespace ode = boost::numeric::odeint;

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



        bool reset(state_type & X) const {
                if (X[0] < params[9])
                        return false;
                else {
                        X[0] = params[7];
                        X[1] += params[8];
                        return true;
                }
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

template<typename Model>
py::array
integrate(Model & model, typename Model::state_type x, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        double t = 0;
        size_t nsteps = floor(tmax / dt) + 1;
        auto obs = spyks::pyarray_writer<Model>(nsteps);
        auto stepper = spyks::integrators::resetting_euler<state_type>();
        ode::integrate_const(stepper, model, x, 0.0, tmax, dt, obs);
        return obs.X;
}

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
            .def("reset", [](adex const & m, adex::state_type & X) {
                            bool r = m.reset(X);
                            return std::make_pair(r, X);
                    });

    m.def("integrate", [](py::array_t<double, py::array::c_style | py::array::forcecast> params,
                          adex::state_type x0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> forcing,
                          double forcing_dt, double stepping_dt) -> py::array {
                  double const * pptr = static_cast<double const *>(params.data());
                  py::buffer_info forcing_info = forcing.request();
                  double const * dptr = static_cast<double const *>(forcing_info.ptr);
                  double tmax = forcing_info.shape[0] * forcing_dt;
                  adex model(pptr, dptr, forcing_dt);
                  return integrate(model, x0, tmax, stepping_dt);
          },
          "Integrates adex model from starting state x0 over the duration of the forcing timeseries",
          "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
    return m.ptr();
}
