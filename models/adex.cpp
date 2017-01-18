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

template<typename T> inline
T const * interpolate(double t, T const * data, double dt, size_t NC)
{
        size_t index = std::round(t / dt);
        return data + index * NC;
}


/** This observer does nothing. It's mostly here for benchmarking */
template <typename Model>
struct noop_observer {
        typedef typename Model::state_type state_type;
        py::array X;
        void operator()(state_type const & x, double time) {}
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

template <typename value_type, typename time_type=double>
struct adex {
        static const size_t N_PARAM = 10;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef typename std::array<value_type, N_STATE> state_type;
        value_type const * p;
        value_type const * forcing;
        time_type dt;

        adex (value_type const * p, value_type const * f, time_type forcing_dt)
             : p(p), forcing(f), dt(forcing_dt) {}

        void operator()(state_type const & X,
                        state_type & dXdt,
                        time_type t) const {
                const double Iinj = interpolate(t, forcing, dt, N_FORCING)[0];
                const double x0 = -X[1];
const double x1 = X[0] - p[2];
                dXdt[0] = (Iinj + x0 - x1*p[1] + exp((X[0] - p[4])/p[3])*p[1]*p[3])/p[0];
dXdt[1] = (x0 + x1*p[6])/p[5];
        }

        bool reset(state_type & X) const {
                bool rp = X[0] >= p[9] ;
                if (rp) {
                        X[0] = p[7];
X[1] = X[1] + p[8];
                }
                return rp;
        }

        void clip(state_type & X) const {
                if (X[0] > p[9]) X[0] = p[9];
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

using spyks::adex;

PYBIND11_PLUGIN(adex) {
        typedef double value_type;
        typedef double time_type;
        typedef adex<value_type, time_type> model;
        py::module m("adex", "adaptive exponential integrate and fire model");
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
