
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
