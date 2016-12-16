#ifndef NEURONS_H
#define NEURONS_H

#include <array>
#include <cmath>

namespace spyks { namespace neurons {

/**
 * timeseries is a thin wrapper for a 1- or 2-D array that represents a
 * univariate or multivariate timeseries. It provides basic nearest-neighbor
 * interpolation through operator(). It does not own its data and should only be
 * used as a temporary/local variable.
 */
class timeseries {
public:
        timeseries(double * data, size_t NC, size_t NT, double dt)
                : _ptr(data), _NC(NC), _NT(NT), _dt(dt) {}
        double & operator()(size_t j, double t) {
                return _ptr[j + _NC * index(t)];
        }
        double operator()(size_t j, double t) const {
                return _ptr[j + _NC * index(t)];
        }
        double duration() const {
                return _NT * _dt;
        }
        size_t index(double t) const {
                return std::round(t / _dt);
        }
        size_t dimension() const {
                return _NC;
        }
        double dt() const {
                return _dt;
        }
private:
        double * _ptr;
        size_t _NC, _NT;
        double _dt;
};


struct adex {
        static const size_t N_PARAM = 11;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;

        adex(double const * parameters, timeseries const & forcing);

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        /** Checks reset conditions are true. If so, clips spike to h. */
        bool check_reset(state_type & X) const;

        /** Executes post-spike reset */
        void reset_state(state_type & X) const;

        double C, gl, el, delt, vt, tw, a, vr, b, h, R;
        timeseries forcing;
};

}} // namespace spykes::neurons


#endif /* NEURONS_H */
