#include "neurons.hpp"
#include <algorithm>
#include <cmath>

using namespace spyks::neurons;

inline double square(double x) { return x * x; }

adex::adex(double const * params, timeseries const & forcing) :
        C(params[0]), gl(params[1]), el(params[2]), delt(params[3]),
        vt(params[4]), tw(params[5]), a(params[6]),
        vr(params[7]), b(params[8]), h(params[9]), R(params[10]), forcing(forcing)
{}


void
adex::operator()(state_type const & X, state_type & dXdt, double t) const
{
        double I = forcing(0, t);
        dXdt[0] = 1/C*(-gl*(X[0]-el) + gl*delt*std::exp((X[0]-vt)/delt) - X[1] + R * I);
        dXdt[1] = 1/tw*(a*(X[0]-el) - X[1]);
}

bool
adex::check_reset(state_type & X) const
{
        if (X[0] < h)
                return false;
        X[0] = h;
        return true;
}

void
adex::reset_state(state_type & X) const
{
        X[0] = vr;
        X[1] += b;
}


nakl::nakl(double const * params, timeseries const & forcing) :
        C(params[0]), gna(params[1]), Ena(params[2]), gk(params[3]), Ek(params[4]), gl(params[5]), El(params[6]),
        vm(params[7]), dvm(params[8]), tm0(params[9]), tm1(params[10]), vmt(params[11]), dvmt(params[12]),
        vh(params[13]), dvh(params[14]), th0(params[15]), th1(params[16]), vht(params[17]), dvht(params[18]),
        vn(params[19]), dvn(params[20]), tn0(params[21]), tn1(params[22]), vnt(params[23]),
        dvnt(params[24]), Isa(params[25]), forcing(forcing)
{}


void
nakl::operator()(state_type const & X, state_type & dXdt, double t) const
{
        const double I = forcing(0, t);
        const double &V = X[0], &m = X[1], &h = X[2], &n = X[3];

        dXdt[0] = 1/C * ((gna*m*m*m*h*(Ena - V)) +
                         (gk*n*n*n*n*(Ek - V)) +
                         (gl*(El-V)) + I/Isa);

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

biocm::biocm(double const * params, timeseries const & forcing) :
        C(params[0]), E_l(params[1]), E_na(params[2]), E_k(params[3]), E_h(params[4]), g_l(params[5]), g_na(params[6]), g_kdr(params[7]), g_ka(params[8]), g_kht(params[9]), g_klt(params[10]), g_hcn(params[11]), nam_v(params[12]), nam_dv(params[13]), nam_tau(params[14]), nah_v(params[15]), nah_dv(params[16]), nah_tau(params[17]), kdrm_v(params[18]), kdrm_dv(params[19]), kdrm_tau(params[20]), kam_v(params[21]), kam_dv(params[22]), kam_p(params[23]), kam_t0(params[24]), kam_t1(params[25]), kam_tv(params[26]), kam_tdv1(params[27]), kam_t2(params[28]), kam_tdv2(params[29]), kah_v(params[30]), kah_dv(params[31]), kah_p(params[32]), kah_t0(params[33]), kah_t1(params[34]), kah_tv(params[35]), kah_tdv1(params[36]), kah_t2(params[37]), kah_tdv2(params[38]), kac_t0(params[39]), kac_t1(params[40]), kac_tv(params[41]), kac_tdv1(params[42]), kltm_v(params[43]), kltm_dv(params[44]), kltm_p(params[45]), kltm_t0(params[46]), kltm_t1(params[47]), kltm_tv(params[48]), kltm_tdv1(params[49]), kltm_t2(params[50]), kltm_tdv2(params[51]), klth_z(params[52]), klth_v(params[53]), klth_dv(params[54]), klth_t0(params[55]), klth_t1(params[56]), klth_tv(params[57]), klth_tdv1(params[58]), klth_t2(params[59]), klth_tdv2(params[60]), kht_phi(params[61]), khtm_v(params[62]), khtm_dv(params[63]), khtm_p(params[64]), khtm_t0(params[65]), khtm_t1(params[66]), khtm_tv(params[67]), khtm_tdv1(params[68]), khtm_t2(params[69]), khtm_tdv2(params[70]), khtn_v(params[71]), khtn_dv(params[72]), khtn_t0(params[73]), khtn_t1(params[74]), khtn_tv(params[75]), khtn_tdv1(params[76]), khtn_t2(params[77]), khtn_tdv2(params[78]), hcnh_v(params[79]), hcnh_dv(params[80]), hcnh_t0(params[81]), hcnh_t1(params[82]), hcnh_tv(params[83]), hcnh_tdv1(params[84]), hcnh_t2(params[85]), hcnh_tdv2(params[86]),
        forcing(forcing)
{}

void
biocm::operator()(state_type const & X, state_type & dXdt, double t) const
{
        double sstate, tau;
        const double &V = X[0], &na_m = X[1], &na_h = X[2],
                &kdr_m = X[3], &ka_m = X[4], &ka_h = X[5],
                &ka_c = X[6], &klt_m = X[7], &klt_h = X[8],
                &kht_m = X[9], &kht_n = X[10], &hcn_h = X[11];
        double  &dV = dXdt[0], &dna_m = dXdt[1], &dna_h = dXdt[2],
                &dkdr_m = dXdt[3], &dka_m = dXdt[4], &dka_h = dXdt[5],
                &dka_c = dXdt[6], &dklt_m = dXdt[7], &dklt_h = dXdt[8],
                &dkht_m = dXdt[9], &dkht_n = dXdt[10], &dhcn_h = dXdt[11];
        const double I = forcing(0, t);

        // dV
        dV = 1/C * (g_l * (E_l - V) +
                    g_na * na_m * na_m * na_m * na_h * (E_na - V) +    // Traub-Miles sodium
                    g_kdr * kdr_m * kdr_m * kdr_m * kdr_m * (E_k - V) + // Traub-Miles delayed rectifier
                    // from Rothman and Manis
                    g_ka * ka_m * ka_m * ka_m * ka_m * ka_h  * ka_c * (E_k - V) +
                    g_klt * klt_m * klt_m * klt_m * klt_m * klt_h * (E_k - V) +
                    g_kht * (kht_phi * kht_m * kht_m + (1 - kht_phi) * kht_n) * (E_k - V) +
                    g_hcn * hcn_h * (E_h + V) +
                    I);
        dna_m = dna_h = dkdr_m = dka_m = dka_h = dka_c = dklt_m = dklt_h = dkht_m = dkht_n = dhcn_h = 0;
        // Na_m
        sstate = 1/(1 + exp(-(V - nam_v) / nam_dv));
        dna_m = (sstate - na_m) / nam_tau;

        // Na_h
        sstate = 1/(1 + exp((V - nah_v) / nah_dv));
        dna_h = (sstate - na_h) / nah_tau;

        // KDR
        sstate = 1/(1 + exp(-(V - kdrm_v) / kdrm_dv));
        dkdr_m = (sstate - kdr_m) / kdrm_tau;

        // KA_m
        sstate = pow(1 + exp(-(V - kam_v) / kam_dv), kam_p);
        tau = kam_t0 + 1/(kam_t1 * exp((V - kam_tv) / kam_tdv1) +
                           kam_t2 * exp(-(V - kam_tv) / kam_tdv2));
        dka_m = (sstate - ka_m) / tau;

        // KA_h
        sstate = pow(1 + exp((V - kah_v) / kah_dv), kah_p);
        tau = kah_t0 + 1/(kah_t1 * exp((V - kah_tv) / kah_tdv1) +
                           kah_t2 * exp(-(V - kah_tv) / kah_tdv2));
        dka_h = (sstate - ka_h) / tau;

        // KA_c (secondary inactivation variable)
        // steady-state is same as h
        tau = kac_t0 + kac_t1/(1 + exp(-(V - kac_tv) / kah_tdv1));
        dka_c = (sstate - ka_c) / tau;

        // KLT_m
        sstate = pow(1 + exp(-(V - kltm_v) / kltm_dv), kltm_p);
        tau = kltm_t0 + 1/(kltm_t1 * exp((V - kltm_tv) / kltm_tdv1) +
                           kltm_t2 * exp(-(V - kltm_tv) / kltm_tdv2));
        dklt_m = (sstate - klt_m) / tau;

        // KLT_h
        sstate = (1 - klth_z) / (1 + exp((V - klth_v) / klth_dv)) + klth_z;
        tau = klth_t0 + 1/(klth_t1 * exp((V - klth_tv) / klth_tdv1) +
                           klth_t2 * exp(-(V - klth_tv) / klth_tdv2));
        dklt_h = (sstate - klt_h) / tau;

        // // KHT_m
        sstate = pow(1 + exp(-(V - khtm_v) / khtm_dv), khtm_p);
        tau = khtm_t0 + 1/(khtm_t1 * exp((V - khtm_tv) / khtm_tdv1) +
                           khtm_t2 * exp(-(V - khtm_tv) / khtm_tdv2));
        dkht_m = (sstate - kht_m) / tau;

        // KHT_n (second activation variable)
        sstate = 1/(1 + exp(-(V - khtn_v) / khtn_dv));
        tau = khtn_t0 + 1/(khtn_t1 * exp((V - khtn_tv) / khtn_tdv1) +
                           khtn_t2 * exp(-(V - khtn_tv) / khtn_tdv2));
        dkht_n = (sstate - kht_n) / tau;

        // HCN
        sstate = 1/(1 + exp((V - hcnh_v) / hcnh_dv));
        tau = hcnh_t0 + 1/(hcnh_t1 * exp((V - hcnh_tv) / hcnh_tdv1) +
                           hcnh_t2 * exp(-(V - hcnh_tv) / hcnh_tdv2));
        dhcn_h = (sstate - hcn_h) / tau;
}
