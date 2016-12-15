#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <boost/numeric/odeint.hpp>
#include "neurons.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
namespace ode = boost::numeric::odeint;
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

/** an observer that writes to a numpy array */
template <typename Model>
struct pyarray_writer {
        typedef typename Model::state_type state_type;
        pyarray_writer(size_t nsteps)
                : step(0),
                  X(py::dtype::of<double>(), {nsteps, Model::N_STATE}),
                  t(py::dtype::of<double>(), {nsteps, 1}) {}
        void operator()(state_type const & x, double time) {
                double * dptr = static_cast<double*>(X.mutable_data(step));
                double * tptr = static_cast<double*>(t.mutable_data(step));
                std::copy_n(x.begin(), Model::N_STATE, dptr);
                *tptr = time;
                ++step;
        }
        size_t step;
        py::array X;
        py::array t;
};


std::pair<py::array, py::array>
integrate_adex(adex const & model, adex::state_type const & x0, double dt)
{
        double t = 0;
        double tspan = model.forcing.duration();
        size_t nsteps = floor(tspan / dt);
        adex::state_type x = x0;
        std::vector<size_t> shape = { nsteps, adex::N_STATE };
        auto obs = pyarray_writer<adex>(nsteps);
        auto stepper = ode::euler<adex::state_type>();
        for (size_t i = 0; i < nsteps; ++i) {
                obs(x, t);
                // if (!model.reset(x))
                //         stepper.do_step(model, x, t, dt);
                t += dt;
        }
        return std::make_pair(obs.X, obs.t.squeeze());
        //
        // m.set_forcing(forcing);
        // while (t < tspan) {
        //         out.write(x, t);
        //         if (!m.reset(x))
        //                 stepper.do_step(m, x, t, dt);
        //         t += dt;
        // }
        // return out;
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
    m.def("integrate_adex", &integrate_adex);
    return m.ptr();
}
