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

struct nakl {
        static const size_t N_PARAM = 26;
        static const size_t N_STATE = 4;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;

        nakl(double const * parameters, timeseries const & forcing);

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        double C, gna, Ena, gk, Ek, gl, El, vm, dvm, tm0, tm1, vmt, dvmt, vh, dvh,
                th0, th1, vht, dvht, vn, dvn, tn0, tn1, vnt, dvnt, Isa;
        timeseries forcing;
};

struct biocm {
        static const size_t N_PARAM = 92;
        static const size_t N_STATE = 12;
        static const size_t N_FORCING = 1;
        typedef std::array<double, N_STATE> state_type;

        biocm(double const * parameters, timeseries const & forcing);

        /** Calculates equations of motion dX/dt = F(X, theta, t) */
        void operator()(state_type const & X, state_type & dXdt, double t) const;

        double C, E_l, E_na, E_k, E_h, g_l, g_na, g_kdr, g_ka, g_kht, g_klt, g_hcn, nam_v, nam_dv, nam_tau, nah_v, nah_dv, nah_t0, nah_t1, nah_tv, nah_tdv1, nah_t2, nah_tdv2, kdrm_v, kdrm_dv, kdrm_tau, kam_v, kam_dv, kam_p, kam_t0, kam_t1, kam_tv, kam_tdv1, kam_t2, kam_tdv2, kah_v, kah_dv, kah_p, kah_t0, kah_t1, kah_tv, kah_tdv1, kah_t2, kah_tdv2, kac_t0, kac_t1, kac_tv, kac_tdv1, kltm_v, kltm_dv, kltm_p, kltm_t0, kltm_t1, kltm_tv, kltm_tdv1, kltm_t2, kltm_tdv2, klth_z, klth_v, klth_dv, klth_t0, klth_t1, klth_tv, klth_tdv1, klth_t2, klth_tdv2, kht_phi, khtm_v, khtm_dv, khtm_p, khtm_t0, khtm_t1, khtm_tv, khtm_tdv1, khtm_t2, khtm_tdv2, khtn_v, khtn_dv, khtn_t0, khtn_t1, khtn_tv, khtn_tdv1, khtn_t2, khtn_tdv2, hcnh_v, hcnh_dv, hcnh_t0, hcnh_t1, hcnh_tv, hcnh_tdv1, hcnh_t2, hcnh_tdv2;
        timeseries forcing;
};

}} // namespace spykes::neurons


#endif /* NEURONS_H */
