
#include "neurons.hpp"
#include <algorithm>
#include <cmath>

using namespace neurons;

void
adex::set_params(parameters_type const params)
{
        std::copy_n(params, N_PARAM, _params.begin());
}

void
adex::set_forcing(forcing_type const & Iinj, double dt)
{
        forcing_type::size_type N = Iinj.size();
        _Iinj.resize(N);
        std::copy(Iinj.begin(), Iinj.end(), _Iinj.begin());
        _dt = dt;
}

void
adex::operator()(state_type const X, state_type dXdt, double t) const
{
        // refs for convenience; will most likely be optimized out
        typedef double const & dcr;
        dcr C = _params[0],
            gl = _params[1],
            el = _params[2],
            delt = _params[3],
            vt = _params[4],
            tw = _params[5],
            a = _params[6],
            R = _params[10];

        forcing_type::size_type rt = std::round(t / _dt);
        double Iinj = _Iinj[rt];
        dXdt[0] = 1/C*(-gl*(X[0]-el) + gl*delt*std::exp((X[0]-vt)/delt) - X[1] + R*Iinj);
        dXdt[1] = 1/tw*(a*(X[0]-el) - X[1]);
}

bool
adex::reset(state_type X) const
{
        typedef double const & dcr;
        dcr vr = _params[7],
            b = _params[8],
            h = _params[9];

        if (X[0] < h)
                return false;
        else {
                X[0] = vr;
                X[1] += b;
                return true;
        }
}
