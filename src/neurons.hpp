#ifndef NEURONS_H
#define NEURONS_H

#include <vector>
#include <map>
#include <array>
#include <algorithm>

namespace neurons {


class adex {
public:
        static const int PARAM_DIM = 11;
        static const int STATE_DIM = 3;
        typedef std::vector<double> state_type;
        typedef std::vector<double> forcing_type;

        adex();

        /** Updates the parameters of the model */
        template <typename It>
        void set_params(It first, size_t N) {
                std::copy_n(first, N, _params);
        }

        /** Returns the current value of the parameters */
        // parameters_t const & get_theta() const;

        /** Sets the forcing terms in the model */
        void set_forcing(forcing_type const &, double dt);

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        /** Resets X if reset conditions are true */
        bool reset(state_type & X) const;

private:
        double _params[PARAM_DIM];
        double & _C, _gl, _el, _delt, _vt, _tw, _a, _vr, _b, _h, _R;
        forcing_type _Iinj;
        double _dt;
};

}

#endif /* NEURONS_H */
