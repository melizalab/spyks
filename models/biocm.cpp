// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/numeric/odeint.hpp>

namespace py = pybind11;
using namespace pybind11::literals;
namespace ode = boost::numeric::odeint;

template <class T>
inline constexpr T pow(T x, std::size_t n){
    return n>0 ? x * pow(x, n - 1):1;
}

namespace spyks {

template <typename value_t, typename time_t>
struct nn_interpolator {
        typedef value_t value_type;
        typedef time_t time_type;
        typedef typename py::array_t<value_type> array_type;

        nn_interpolator(array_type data, time_type dt)
                : data(data), dt(dt), N(data.shape(0)) {}

        template<typename... Ix> value_type operator()(time_type t, Ix... idx) const {
                // TODO avoid numpy bounds check
                return data.at(index_at(t), idx...);
        }
        size_t index_at(time_type t) const {
                if (t < 0) return 0;
                size_t i = std::round(t / dt);
                return std::min(i, N - 1);
        }

        array_type data;
        time_type dt;
        size_t N;
};

template <typename Model>
struct pyarray_dense {
        typedef typename Model::state_type state_type;
        pyarray_dense(size_t nsteps)
                : nsteps(nsteps), step(0),
                  X(py::dtype::of<double>(), {nsteps, Model::N_STATE}) {}
        void operator()(state_type const & x, double time) {
                if (step < nsteps) {
                        double * dptr = static_cast<double *>(X.mutable_data(step));
                        std::copy_n(x.begin(), Model::N_STATE, dptr);
                }
                ++step;
        }
        const size_t nsteps;
        size_t step;
        py::array X;
};

}


namespace spyks {

template <typename T, typename interpolator_type>
struct biocm {
        static const size_t N_PARAM = 88;
        static const size_t N_STATE = 12;
        static const size_t N_FORCING = 1;
        typedef T value_type;
        typedef typename std::array<value_type, N_STATE> state_type;
        typedef typename interpolator_type::time_type time_type;
        value_type const * p;
        interpolator_type forcing;

        biocm (value_type const * p, interpolator_type f)
             : p(p), forcing(f) {}

        void operator()(state_type const & X,
                        state_type & dXdt,
                        time_type t) const {
                const double Iinj = forcing(t);
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

template<typename Model>
py::array
integrate(Model & model, py::array_t<typename Model::value_type> x0, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        state_type x;
        std::copy_n(x0.data(), Model::N_STATE, x.begin());
        size_t nsteps = ceil(tmax / dt);
        auto obs = pyarray_dense<Model>(nsteps);
        auto stepper = ode::runge_kutta_dopri5<state_type>();
        ode::integrate_const(ode::make_dense_output(1.0e-4, 1.0e-4, stepper),
                             std::ref(model), x, 0.0, tmax, dt, obs);
        return obs.X;
}

}


PYBIND11_PLUGIN(biocm) {
        typedef double value_type;
        typedef double time_type;
        typedef spyks::nn_interpolator<value_type, time_type> interpolator;
        typedef spyks::biocm<value_type, interpolator> model;
        py::module m("biocm", "biophysical model of CM excitatory neuron");
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
