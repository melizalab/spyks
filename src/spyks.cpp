#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/numeric/odeint.hpp>
#include "neurons.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace neurons;

// py::array_t<double>
// integrate_adex(adex const & m, double dt)
// {
//         auto stepper = euler<adex::state_type>();



PYBIND11_PLUGIN(models) {
    py::module m("models", "spiking neuron models");
    py::class_<adex>(m, "AdEx")
            .def(py::init<>())
            .def("set_params", [](adex &m, py::array_t<double, py::array::c_style | py::array::forcecast> params) {
                            py::buffer_info info = params.request();
                            double * ptr = static_cast<double *>(info.ptr);
                            m.set_params(ptr);
                    })
            .def("set_forcing", &adex::set_forcing)
            .def("__call__", [](adex const & m, adex::state_type const & X, double t) -> adex::state_type {
                            adex::state_type out;
                            m(X, out, t);
                            return out;
                    })
            .def("reset", [](adex const & m, adex::state_type & X) -> std::pair<bool, adex::state_type> {
                            bool r = m.reset(X);
                            return make_pair(r, X);
                    });
    return m.ptr();
}
