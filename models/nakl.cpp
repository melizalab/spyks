#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "utility.hpp"
#include "integrators.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
namespace ode = boost::numeric::odeint;

namespace spyks { namespace models {

struct nakl {
        static const size_t N_PARAM = 25;
        static const size_t N_STATE = 4;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;
        double const * params;
        double const * forcing;
        double dt;

        nakl(double const * parameters, double const * forcing, double forcing_dt)
                : params(parameters), forcing(forcing), dt(forcing_dt) {}

        void operator()(state_type const & X, state_type & dXdt, double t) const {
                const double &C(params[0]), &gna(params[1]), &Ena(params[2]), &gk(params[3]),
                        &Ek(params[4]), &gl(params[5]), &El(params[6]),
                        &vm(params[7]), &dvm(params[8]), &tm0(params[9]),
                        &tm1(params[10]), &vmt(params[11]), &dvmt(params[12]),
                        &vh(params[13]), &dvh(params[14]), &th0(params[15]),
                        &th1(params[16]), &vht(params[17]), &dvht(params[18]),
                        &vn(params[19]), &dvn(params[20]), &tn0(params[21]),
                        &tn1(params[22]), &vnt(params[23]), &dvnt(params[24]);
                const double &V = X[0], &m = X[1], &h = X[2], &n = X[3];
                const double I = interpolate(t, forcing, dt, N_FORCING)[0];
                dXdt[0] = 1/C * ((gna*m*m*m*h*(Ena - V)) +
                                 (gk*n*n*n*n*(Ek - V)) +
                                 (gl*(El-V)) + I);

                double taum = tm0 + tm1 * (1-square(tanh((V - vmt)/dvmt)));
                double m0 = (1+tanh((V - vm)/dvm))/2;
                dXdt[1] = (m0 - m)/taum;

                double tauh = th0 + th1 * (1-square(tanh((V - vht)/dvht)));
                double h0 = (1+tanh((V - vh)/dvh))/2;
                dXdt[2] = (h0 - h)/tauh;

                double taun = tn0 + tn1 * (1-square(tanh((V - vnt)/dvnt)));
                double n0 = (1+tanh((V - vn)/dvn))/2;
                dXdt[3] = (n0 - n)/taun;
        }
};

}}

template<typename Model>
py::array
integrate(Model & model, typename Model::state_type x, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        double t = 0;
        size_t nsteps = floor(tmax / dt) + 1;
        auto obs = spyks::pyarray_writer<Model>(nsteps);
        auto stepper = ode::runge_kutta_dopri5<state_type>();
        ode::integrate_const(ode::make_dense_output(1.0e-4, 1.0e-4, stepper),
                             std::ref(model), x, 0.0, tmax, dt, obs);
        return obs.X;
}

using spyks::models::nakl;

PYBIND11_PLUGIN(nakl) {
    py::module m("nakl", "adaptive exponential model");
    py::class_<nakl>(m, "model")
            .def("__init__", [](nakl &m,
                                py::array_t<double, py::array::c_style | py::array::forcecast> params,
                                py::array_t<double, py::array::c_style | py::array::forcecast> forcing,
                                double forcing_dt) {
                         auto pptr = static_cast<double const *>(params.data());
                         auto dptr = static_cast<double const *>(forcing.data());
                         new (&m) nakl(pptr, dptr, forcing_dt);
                 })
            .def("__call__", [](nakl const & m, nakl::state_type const & X, double t) -> nakl::state_type {
                            nakl::state_type out;
                            m(X, out, t);
                            return out;
                    });
    m.def("integrate", [](py::array_t<double, py::array::c_style | py::array::forcecast> params,
                          nakl::state_type x0,
                          py::array_t<double, py::array::c_style | py::array::forcecast> forcing,
                          double forcing_dt, double stepping_dt) -> py::array {
                  double const * pptr = static_cast<double const *>(params.data());
                  py::buffer_info forcing_info = forcing.request();
                  double const * dptr = static_cast<double const *>(forcing_info.ptr);
                  double tmax = forcing_info.shape[0] * forcing_dt;
                  nakl model(pptr, dptr, forcing_dt);
                  return integrate(model, x0, tmax, stepping_dt);
          },
          "Integrates model from starting state x0 over the duration of the forcing timeseries",
          "params"_a, "x0"_a, "forcing"_a, "forcing_dt"_a, "stepping_dt"_a);
    return m.ptr();
}
