#ifndef INTEGRATORS_H
#define INTEGRATORS_H

#include <boost/numeric/odeint.hpp>

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

        bool in_reset;

        resetting_euler() : in_reset(false) {}

        template <typename System>
        void do_step(System system, state_type & x, time_type t, time_type dt) {
                deriv_type dxdt;
                if (in_reset) {
                        system.reset_state(x);
                        in_reset = false;
                }
                else {
                        system(x, dxdt, t);
                        for (size_t i = 0; i < x.size(); ++i)
                                x[i] += dt * dxdt[i];
                        in_reset = system.check_reset(x);
                }
        }
};


}}  // namespace spyks::integrators

#endif /* INTEGRATORS_H */
