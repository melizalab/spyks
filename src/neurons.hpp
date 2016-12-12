#ifndef NEURONS_H
#define NEURONS_H

#include <vector>
#include <map>
#include <array>
#include <algorithm>

/*
 * General notes about the C++ API
 *
 */

namespace neurons {


class adex {
public:
        static const size_t N_PARAM = 11;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef std::vector<double> parameters_type;
        typedef std::vector<double> state_type;
        typedef std::vector<double> forcing_type;

        /** Updates the parameters of the model */
        void set_params(parameters_type const &);

        parameters_type const & get_params() const;
        /** Returns the current value of the parameters */
        // parameters_t const & get_theta() const;

        /** Sets the forcing terms in the model */
        void set_forcing(forcing_type const &, double dt);

        forcing_type const & get_forcing() const;

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        /** Resets X if reset conditions are true */
        bool reset(state_type & X) const;

private:
        std::array<double, N_PARAM> _params;
        forcing_type _Iinj;
        double _dt;
};

}

#endif /* NEURONS_H */
