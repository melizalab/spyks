// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "spyks/integrators.h"

namespace py = pybind11;
using namespace pybind11::literals;

template <class T>
inline constexpr T pow(T x, std::size_t n){
    return n>0 ? x * pow(x, n - 1):1;
}

namespace spyks { namespace models {

template <typename value_type, typename time_type=double>
struct biocm {
        static const size_t N_PARAM = 88;
        static const size_t N_STATE = 12;
        static const size_t N_FORCING = 1;
        typedef typename std::array<value_type, N_STATE> state_type;
        value_type const * p;
        value_type const * forcing;
        time_type dt;

        biocm (value_type const * p, value_type const * f, time_type forcing_dt)
             : p(p), forcing(f), dt(forcing_dt) {}

        void operator()(state_type const & X,
                        state_type & dXdt,
                        time_type t) const {
                const double Iinj = interpolate(t, forcing, dt, N_FORCING)[0];

                const double x0 = -X[0];
const double x1 = x0 + p[3];
const double x2 = exp((x0 + p[12])/p[13]) + 1;
const double x3 = X[0] - p[19];
const double x4 = exp((x0 + p[23])/p[24]) + 1;
const double x5 = X[0] - p[30];
const double x6 = -1/sqrt(exp((X[0] - p[34])/p[35]) + 1);
const double x7 = X[0] - p[38];
const double x8 = X[0] - p[50];
const double x9 = X[0] - p[59];
const double x10 = X[0] - p[68];
const double x11 = X[0] - p[76];
const double x12 = X[0] - p[84];

dXdt[0] = (Iinj + x1*(-(p[63] - 1)*X[10] + pow(X[9], 2)*p[63])*p[9] + x1*pow(X[3], 4)*p[7] + x1*pow(X[4], 4)*X[5]*X[6]*p[8] + x1*pow(X[7], 4)*X[8]*p[10] + (x0 + p[1])*p[5] + (x0 + p[2])*pow(X[1], 3)*X[2]*p[6] + (x0 + p[4])*X[11]*p[11])/p[0];
dXdt[1] = (-x2*X[1] + 1)/(x2*p[14]);
dXdt[2] = -(X[2] - 1/(exp((X[0] - p[15])/p[16]) + 1))/(p[17] + 1.0/(exp(x3/p[20])*p[18] + exp(-x3/p[22])*p[21]));
dXdt[3] = (-x4*X[3] + 1)/(x4*p[25]);
dXdt[4] = -(X[4] - 1/pow(1 + exp(-(X[0] - p[26])/p[27]), 1.0L/4.0L))/(p[28] + 1.0/(exp(x5/p[31])*p[29] + exp(-x5/p[33])*p[32]));
dXdt[5] = -(x6 + X[5])/(p[36] + 1.0/(exp(x7/p[39])*p[37] + exp(-x7/p[41])*p[40]));
dXdt[6] = -(x6 + X[6])/(p[42] + p[43]/(1 + exp(-(X[0] - p[44])/p[45])));
dXdt[7] = -(X[7] - 1/pow(1 + exp(-(X[0] - p[46])/p[47]), 1.0L/4.0L))/(p[48] + 1.0/(exp(x8/p[51])*p[49] + exp(-x8/p[53])*p[52]));
dXdt[8] = -(X[8] - p[54] + (p[54] - 1)/(exp((X[0] - p[55])/p[56]) + 1))/(p[57] + 1.0/(exp(x9/p[60])*p[58] + exp(-x9/p[62])*p[61]));
dXdt[9] = -(X[9] - 1/sqrt(1 + exp(-(X[0] - p[64])/p[65])))/(p[66] + 1.0/(exp(x10/p[69])*p[67] + exp(-x10/p[71])*p[70]));
dXdt[10] = -(X[10] - 1/(1 + exp(-(X[0] - p[72])/p[73])))/(p[74] + 1.0/(exp(x11/p[77])*p[75] + exp(-x11/p[79])*p[78]));
dXdt[11] = -(X[11] - 1/(exp((X[0] - p[80])/p[81]) + 1))/(p[82] + 1.0/(exp(x12/p[85])*p[83] + exp(-x12/p[87])*p[86]));
        }
};

}}

using spyks::models::biocm;

PYBIND11_PLUGIN(biocm) {
        typedef double value_type;
        typedef double time_type;
        typedef biocm<value_type, time_type> model;
        py::module m("biocm", "biophysical model of CM excitatory neuron");
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
