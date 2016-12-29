#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "neurons.hpp"
#include "integrators.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace spyks;
using namespace spyks::neurons;
namespace ode = boost::numeric::odeint;


template<typename Model, template<typename> class Stepper >
py::array
integrate(Model & model, typename Model::state_type x, double dt)
{
        typedef typename Model::state_type state_type;
        double t = 0;
        double tspan = model.forcing.duration();
        size_t nsteps = floor(tspan / dt) + 1;

        auto obs = pyarray_writer<Model>(nsteps);
        auto stepper = Stepper<state_type>();
        // can't use std::ref on the model b/c the resetting stepper needs to
        // call several methods
        ode::integrate_const(stepper, model, x, 0.0, tspan, dt, obs);
        return obs.X;
}

template<typename Model>
py::array
integrate(Model & model, typename Model::state_type x, double dt)
{
        typedef typename Model::state_type state_type;
        double t = 0;
        double tspan = model.forcing.duration();
        size_t nsteps = floor(tspan / dt) + 1;

        auto obs = pyarray_writer<Model>(nsteps);
        auto stepper = ode::runge_kutta_dopri5<state_type>();
        ode::integrate_const(ode::make_dense_output(1.0e-4, 1.0e-4, stepper),
                             std::ref(model), x, 0.0, tspan, dt, obs);
        return obs.X;
}


PYBIND11_PLUGIN(models) {
    py::module m("models", "spiking neuron models");
    py::class_<timeseries >(m, "timeseries")
            .def("__init__", [](timeseries & instance,
                                py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                double dt) {
                         py::buffer_info info = data.request();
                         double * ptr = static_cast<double *>(info.ptr);
                         if (info.ndim == 1)
                                 new (&instance) timeseries(ptr, 1, info.shape[0], dt);
                         else if (info.ndim == 2)
                                 new (&instance) timeseries(ptr, info.shape[1], info.shape[0], dt);
                         else
                                 throw std::runtime_error("Incompatible array dimensions (> 2)");
                 })
            .def("duration", &timeseries::duration)
            .def("dimension", &timeseries::dimension)
            .def("dt", &timeseries::dt)
            .def("__call__", [](timeseries & instance, size_t ij, double t) -> double {
                            return instance(ij, t);
                    });

    py::class_<adex>(m, "AdEx")
            .def("__init__", [](adex &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                timeseries & forcing) {
                         auto pptr = static_cast<double const *>(params.data());
                         new (&m) adex(pptr, forcing);
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

    py::class_<nakl>(m, "NaKL")
            .def("__init__", [](nakl &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                timeseries & forcing) {
                         auto pptr = static_cast<double const *>(params.data());
                         new (&m) nakl(pptr, forcing);
                 })
            .def("__call__", [](nakl const & m, nakl::state_type const & X, double t) -> nakl::state_type {
                            nakl::state_type out;
                            m(X, out, t);
                            return out;
                    });
    py::class_<biocm>(m, "biocm")
            .def("__init__", [](biocm &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                timeseries & forcing) {
                         auto pptr = static_cast<double const *>(params.data());
                         new (&m) biocm(pptr, forcing);
                 })
            .def("__call__", [](biocm const & m, biocm::state_type const & X, double t) -> biocm::state_type {
                            biocm::state_type out;
                            m(X, out, t);
                            return out;
                    });
    m.def("integrate", &integrate<adex, integrators::resetting_euler>);
    m.def("integrate", &integrate<nakl>);
    m.def("integrate", &integrate<biocm>);
    return m.ptr();
}
