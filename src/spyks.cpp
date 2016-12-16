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

/** an observer that writes to a numpy array */
template <typename Model>
struct pyarray_writer {
        typedef typename Model::state_type state_type;
        pyarray_writer(size_t nsteps)
                : step(0), X(py::dtype::of<double>(), {nsteps, Model::N_STATE}) {}
        void operator()(state_type const & x, double time) {
                double * dptr = static_cast<double*>(X.mutable_data(step));
                std::copy_n(x.begin(), Model::N_STATE, dptr);
                ++step;
        }
        size_t step;
        py::array X;
};

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
        ode::integrate_const(stepper, model, x, 0.0, tspan, dt, obs);
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
    m.def("integrate_adex", &integrate<adex, integrators::resetting_euler>);
    return m.ptr();
}
