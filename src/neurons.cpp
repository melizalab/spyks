
#include "neurons.hpp"
#include <algorithm>
#include <cmath>

using namespace neurons;

adex::adex(double const * params, forcing_type const & forcing) :
        _C(params[0]), _gl(params[1]), _el(params[2]), _delt(params[3]),
        _vt(params[4]), _tw(params[5]), _a(params[6]),
        _vr(params[7]), _b(params[8]), _h(params[9]), _R(params[10]), _forcing(forcing)
{}


void
adex::operator()(state_type const & X, state_type & dXdt, double t) const
{
        double Iinj = _forcing(0, t);
        dXdt[0] = 1/_C*(-_gl*(X[0]-_el) + _gl*_delt*std::exp((X[0]-_vt)/_delt) - X[1] + _R*Iinj);
        dXdt[1] = 1/_tw*(_a*(X[0]-_el) - X[1]);
}

bool
adex::reset(state_type & X) const
{
        if (X[0] < _h)
                return false;
        else {
                X[0] = _vr;
                X[1] += _b;
                return true;
        }
}
