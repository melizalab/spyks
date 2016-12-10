
#include "neurons.hpp"
#include <cmath>

using namespace neurons;

adex::adex()
        : _C(_params[0]), _gl(_params[1]), _el(_params[2]), _delt(_params[3]),
          _vt(_params[4]), _tw(_params[5]), _a(_params[6]), _vr(_params[7]),
          _b(_params[8]), _h(_params[9]), _R(_params[10])
{}

void
adex::set_forcing(forcing_type const & Iinj, double dt)
{
        forcing_type::size_type N = Iinj.size();
        _Iinj.resize(N);
        std::copy(Iinj.begin(), Iinj.end(), _Iinj.begin());
        _dt = dt;
}

void
adex::operator()(state_type const & X, state_type & dXdt, double t) const
{
        forcing_type::size_type rt = std::round(t) / _dt;
        double Iinj = _Iinj[rt];
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
