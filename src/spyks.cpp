#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//#include <boost/numeric/odeint.hpp>
#include "neurons.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace neurons;


template <typename Buffer>
double * get_data(Buffer & b)
{
        py::buffer_info info = b.request();
        return static_cast<double *>(info.ptr);
}
// template<typename Model>
// integrate_reset(py::array_t<double, py::array::c_style | py::array::forcecast> params,
//                 typename Model::forcing_type const & forcing,
//                 typename Model::state_type const & x0, double dt)
// {
//         double t = 0;
//         double tspan = forcing.duration();
//         Model::state_type x;
//         timeseries out(tspan, dt);
//         auto stepper = euler<adex::state_type>();
//         m.set_forcing(forcing);
//         while (t < tspan) {
//                 out.write(x, t);
//                 if (!m.reset(x))
//                         stepper.do_step(m, x, t, dt);
//                 t += dt;
//         }
//         return out;
// }

py::array
integrate_adex(py::array_t<double, py::array::c_style | py::array::forcecast> params,
               adex::forcing_type const & forcing,
               adex::state_type const & x0, double dt)
{
        double t = 0;
        double tspan = forcing.duration();
        size_t nsteps = floor(tspan / dt);
        // adex::state_type x;
        // adex model(params, forcing);
        std::vector<size_t> shape = { adex::N_STATE, nsteps};
        py::array out(py::dtype("d"), shape);
        return out;
        // auto stepper = euler<adex::state_type>();
        // m.set_forcing(forcing);
        // while (t < tspan) {
        //         out.write(x, t);
        //         if (!m.reset(x))
        //                 stepper.do_step(m, x, t, dt);
        //         t += dt;
        // }
        // return out;
}

std::vector<double>
make_vector(std::vector<double> const & blarg)
{
        return blarg;
}


PYBIND11_PLUGIN(models) {
    py::module m("models", "spiking neuron models");
    py::class_<timeseries<double> >(m, "timeseries")
            .def("__init__", [](timeseries<double> & instance,
                                py::array_t<double, py::array::c_style | py::array::forcecast> data,
                                double dt) {
                         py::buffer_info info = data.request();
                         double * ptr = static_cast<double *>(info.ptr);
                         new (&instance) timeseries<double>(ptr, info.shape[0], info.shape[1], dt);
                 })
            .def("duration", &timeseries<double>::duration)
            .def("__call__", [](timeseries<double> & instance, size_t ij, double t) -> double {
                            return instance(ij, t);
                    });

    py::class_<adex>(m, "AdEx")
            .def("__init__", [](adex &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                adex::forcing_type const & forcing) {
                         new (&m) adex(get_data(params), forcing);
                 })
            .def("__call__", [](adex const & m, adex::state_type const & X, double t) -> adex::state_type {
                            adex::state_type out;
                            m(X, out, t);
                            return out;
                    })
            .def("reset", [](adex const & m, adex::state_type & X) -> std::pair<bool, adex::state_type> {
                            bool r = m.reset(X);
                            return make_pair(r, X);
                    });
    m.def("integrate_adex", &integrate_adex);
    m.def("make_vector", &make_vector);
    return m.ptr();
}
