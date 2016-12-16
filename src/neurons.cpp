
#include "neurons.hpp"
#include <algorithm>
#include <cmath>

using namespace spyks::neurons;

adex::adex(double const * params, timeseries const & forcing) :
        C(params[0]), gl(params[1]), el(params[2]), delt(params[3]),
        vt(params[4]), tw(params[5]), a(params[6]),
        vr(params[7]), b(params[8]), h(params[9]), R(params[10]), forcing(forcing)
{}


void
adex::operator()(state_type const & X, state_type & dXdt, double t) const
{
        double Iinj = forcing(0, t);
        dXdt[0] = 1/C*(-gl*(X[0]-el) + gl*delt*std::exp((X[0]-vt)/delt) - X[1] + R*Iinj);
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
