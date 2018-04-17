// -*- coding: utf-8 -*-
// -*- mode: c++ -*-
// automatically generated by spyks, version 0.6.7
// model: biocm_ei
// version: 1.1
// description: model for excitatory CM neurons, based on Rothman and Manis (2003); driving conductances
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
struct biocm_ei {
        static const size_t N_PARAM = 91;
        static const size_t N_STATE = 11;
        static const size_t N_FORCING = 2;
        typedef T value_type;
        typedef typename std::array<value_type, N_STATE> state_type;
        typedef typename interpolator_type::time_type time_type;
        value_type const * p;
        interpolator_type forcing;

        biocm_ei (value_type const * p, interpolator_type f)
             : p(p), forcing(f) {}

        void operator()(state_type const & X,
                        state_type & dXdt,
                        time_type t) const {
                const double g_ex = forcing(t, 0);
const double g_inh = forcing(t, 1);
                const double x0 = -X[0];
const double x1 = x0 + p[3];
const double x2 = X[0] - p[17];
const double x3 = X[0] - p[25];
const double x4 = X[0] - p[71];
const double x5 = X[0] - p[79];
const double x6 = X[0] - p[33];
const double x7 = -1/sqrt(exp((X[0] - p[37])/p[38]) + 1);
const double x8 = X[0] - p[41];
const double x9 = X[0] - p[53];
const double x10 = X[0] - p[62];
const double x11 = X[0] - p[87];
                dXdt[0] = (g_ex*(x0 + p[5]) + g_inh*(x0 + p[6]) + x1*(-(p[66] - 1)*X[4] + pow(X[3], 2)*p[66])*p[10] + x1*pow(X[5], 4)*X[6]*X[7]*p[9] + x1*pow(X[8], 4)*X[9]*p[11] + (x0 + p[1])*p[7] + (x0 + p[2])*pow(X[1], 3)*X[2]*p[8] + (x0 + p[4])*X[10]*p[12])/p[0];
dXdt[1] = -(X[1] - 1/(1 + exp(-(X[0] - p[13])/p[14])))/(p[15] + 1.0/(exp(x2/p[18])*p[16] + exp(-x2/p[20])*p[19]));
dXdt[2] = -(X[2] - 1/(exp((X[0] - p[21])/p[22]) + 1))/(p[23] + 1.0/(exp(x3/p[26])*p[24] + exp(-x3/p[28])*p[27]));
dXdt[3] = -(X[3] - 1/sqrt(1 + exp(-(X[0] - p[67])/p[68])))/(p[69] + 1.0/(exp(x4/p[72])*p[70] + exp(-x4/p[74])*p[73]));
dXdt[4] = -(X[4] - 1/(1 + exp(-(X[0] - p[75])/p[76])))/(p[77] + 1.0/(exp(x5/p[80])*p[78] + exp(-x5/p[82])*p[81]));
dXdt[5] = -(X[5] - 1/pow(1 + exp(-(X[0] - p[29])/p[30]), 1.0L/4.0L))/(p[31] + 1.0/(exp(x6/p[34])*p[32] + exp(-x6/p[36])*p[35]));
dXdt[6] = -(x7 + X[6])/(p[39] + 1.0/(exp(x8/p[42])*p[40] + exp(-x8/p[44])*p[43]));
dXdt[7] = -(x7 + X[7])/(p[45] + p[46]/(1 + exp(-(X[0] - p[47])/p[48])));
dXdt[8] = -(X[8] - 1/pow(1 + exp(-(X[0] - p[49])/p[50]), 1.0L/4.0L))/(p[51] + 1.0/(exp(x9/p[54])*p[52] + exp(-x9/p[56])*p[55]));
dXdt[9] = -(X[9] - p[57] + (p[57] - 1)/(exp((X[0] - p[58])/p[59]) + 1))/(p[60] + 1.0/(exp(x10/p[63])*p[61] + exp(-x10/p[65])*p[64]));
dXdt[10] = -(X[10] - 1/(exp((X[0] - p[83])/p[84]) + 1))/(p[85] + 1.0/(exp(x11/p[88])*p[86] + exp(-x11/p[90])*p[89]));
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


PYBIND11_MODULE(biocm_ei, m) {
        typedef double value_type;
        typedef double time_type;
        typedef spyks::nn_interpolator<value_type, time_type> interpolator;
        typedef spyks::biocm_ei<value_type, interpolator> model;
        m.doc() = "model for excitatory CM neurons, based on Rothman and Manis (2003); driving conductances";
        m.attr("name") = py::cast("biocm_ei");
        m.attr("__version__") = py::cast(1.1);
        py::class_<model>(m, "model")
                .def("__init__",
                     [](model &m,
                        py::array_t<value_type> params,
                        py::array_t<value_type> forcing,
                        time_type forcing_dt) {
                             // TODO: check forcing dimensions and shape
                             auto pptr = static_cast<value_type const *>(params.data());
                             auto _forcing = interpolator(forcing, forcing_dt);
                             new (&m) model(pptr, _forcing);
                     })
                .def("__call__", [](model const & m, model::state_type const & X, time_type t) {
                                model::state_type out;
                                m(X, out, t);
                                return out;
                        });
        m.def("integrate", [](py::array_t<value_type> params,
                              py::array_t<value_type> x0,
                              py::array_t<value_type> forcing,
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
}
