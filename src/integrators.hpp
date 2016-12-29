#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include "utility.hpp"
#include <boost/numeric/odeint.hpp>


namespace ode = boost::numeric::odeint;

namespace spyks { namespace integrators {

template <typename state_type>
class resetting_euler {
public:

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
                }
        }
};

template<typename Model>
py::array
integrate_reset(Model & model, typename Model::state_type x, double tmax, double dt)
{
        typedef typename Model::state_type state_type;
        double t = 0;
        size_t nsteps = floor(tmax / dt) + 1;
        auto obs = spyks::pyarray_writer<Model>(nsteps);
        auto stepper = resetting_euler<state_type>();
        ode::integrate_const(stepper, model, x, 0.0, tmax, dt, obs);
        return obs.X;
}

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

}}  // namespace spyks::integrators

#endif /* INTEGRATORS_H */
