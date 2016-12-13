#ifndef NEURONS_H
#define NEURONS_H

#include <vector>
#include <map>
#include <array>
#include <cmath>

namespace neurons {

template<typename T>
class timeseries {
public:
        timeseries(T * data, size_t NJ, size_t NT, double dt) : _data(data), _NJ(NJ), _NT(NT), _dt(dt) {}
        T & operator()(size_t ij, double t) {
                return _data[ij * _NJ + index(t)];
        }
        T operator()(size_t ij, double t) const {
                return _data[ij * _NJ + index(t)];
        }
        double duration() const {
                return _NT * _dt;
        }
        size_t index(double t) const {
                return std::round(t / _dt);
        }
private:
        T * _data;
        size_t _NJ, _NT;
        double _dt;
};

class adex {
public:
        static const size_t N_PARAM = 11;
        static const size_t N_STATE = 2;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;
        typedef timeseries<double> forcing_type;

        adex(double const * parameters, forcing_type const & forcing);

        /** Updates the parameters of the model */
        // void set_params(parameters_type const);

        /** Sets the forcing terms in the model */
        // void set_forcing(forcing_type const &, double dt);

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        /** Resets X if reset conditions are true */
        bool reset(state_type & X) const;

private:
        double _C, _gl, _el, _delt, _vt, _tw, _a, _vr, _b, _h, _R;
        forcing_type const & _forcing;
};

}


#endif /* NEURONS_H */
