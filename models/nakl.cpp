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
struct nakl {
        static const size_t N_PARAM = 25;
        static const size_t N_STATE = 4;
        static const size_t N_FORCING = 1;
        typedef typename std::array<value_type, N_STATE> state_type;
        value_type const * params;
        value_type const * forcing;
        time_type dt;

        nakl(value_type const * parameters, value_type const * forcing, time_type forcing_dt)
                : params(parameters), forcing(forcing), dt(forcing_dt) {}

        void operator()(state_type const & X, state_type & dXdt, time_type t) const {
                const value_type &C(params[0]), &gna(params[1]), &Ena(params[2]), &gk(params[3]),
                        &Ek(params[4]), &gl(params[5]), &El(params[6]),
                        &vm(params[7]), &dvm(params[8]), &tm0(params[9]),
                        &tm1(params[10]), &vmt(params[11]), &dvmt(params[12]),
                        &vh(params[13]), &dvh(params[14]), &th0(params[15]),
                        &th1(params[16]), &vht(params[17]), &dvht(params[18]),
                        &vn(params[19]), &dvn(params[20]), &tn0(params[21]),
                        &tn1(params[22]), &vnt(params[23]), &dvnt(params[24]);
                const value_type &V = X[0], &m = X[1], &h = X[2], &n = X[3];
                const value_type I = interpolate(t, forcing, dt, N_FORCING)[0];
                dXdt[0] = 1/C * ((gna*m*m*m*h*(Ena - V)) +
                                 (gk*n*n*n*n*(Ek - V)) +
                                 (gl*(El-V)) + I);

                value_type taum = tm0 + tm1 * (1-square(tanh((V - vmt)/dvmt)));
                value_type m0 = (1+tanh((V - vm)/dvm))/2;
                dXdt[1] = (m0 - m)/taum;

                value_type tauh = th0 + th1 * (1-square(tanh((V - vht)/dvht)));
                value_type h0 = (1+tanh((V - vh)/dvh))/2;
                dXdt[2] = (h0 - h)/tauh;

                value_type taun = tn0 + tn1 * (1-square(tanh((V - vnt)/dvnt)));
                value_type n0 = (1+tanh((V - vn)/dvn))/2;
                dXdt[3] = (n0 - n)/taun;
        }
};

}}

using spyks::models::nakl;

PYBIND11_PLUGIN(nakl) {
        typedef double value_type;
        typedef double time_type;
        typedef nakl<value_type, time_type> dnakl;
        py::module m("nakl", "adaptive exponential model");
        py::class_<dnakl>(m, "model")
                .def("__init__",
                     [](dnakl &m,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                        py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                        time_type forcing_dt) {
                             auto pptr = static_cast<value_type const *>(params.data());
                             auto dptr = static_cast<value_type const *>(forcing.data());
                             new (&m) dnakl(pptr, dptr, forcing_dt);
                     })
                .def("__call__", [](dnakl const & m, dnakl::state_type const & X, time_type t) {
                                dnakl::state_type out;
                                m(X, out, t);
                                return out;
                        });
        m.def("integrate", [](py::array_t<value_type, py::array::c_style | py::array::forcecast> params,
                              dnakl::state_type x0,
                              py::array_t<value_type, py::array::c_style | py::array::forcecast> forcing,
                              time_type forcing_dt, time_type stepping_dt) -> py::array {
                      auto pptr = static_cast<value_type const *>(params.data());
                      py::buffer_info forcing_info = forcing.request();
                      auto dptr = static_cast<value_type const *>(forcing_info.ptr);
                      time_type tmax = forcing_info.shape[0] * forcing_dt;
                      dnakl model(pptr, dptr, forcing_dt);
                      return spyks::integrate(model, x0, tmax, stepping_dt);
              },
              "Integrates model from starting state x0 over the duration of the forcing timeseries",
              "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
        m.def("integrate", &spyks::integrate<dnakl>);
        return m.ptr();
}
