#include <vector>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <string>
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <algorithm>

using namespace std;
using namespace boost::numeric::odeint;
namespace py = boost::python;

typedef double dtype;
typedef std::vector<dtype> vtype;

class neuron 
{
    public:
            dtype res;
            vtype iapp; // applied current
            void apply_current(py::numeric::array newcurr, dtype nres) {
                    iapp.clear();
                    res = nres;
                    for (int i = 0; i < len(newcurr); ++i) {
                            iapp.push_back(py::extract<dtype>(newcurr[i]));
                    }
            }
            py::object integrate(boost::function<void(const vtype&, vtype&, dtype)> ode, int ndim, dtype tspan, dtype dt, vtype x) {
                    int currpoint = 0;
                    int gridlength = tspan/dt;

                    int nsteps = 0;
                    int maxsteps = gridlength * 20;

                    npy_intp size[2];
                    size[0] = gridlength;
                    size[1] = ndim;

                    dtype data[gridlength][ndim];
                    
    				runge_kutta_cash_karp54< vtype > stepper;

                    for (int i = 0; i < gridlength; i++)
                    {
                            for (int j = 0; j < ndim; j++)
                            {
                                    //if (x[j] != x[j]) throw runtime_error( "NaN");
                                    data[i][j] = x[j];
                            }

                            	stepper.do_step(ode,x,i*dt,dt);
          
                    }


                    PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                    py::handle<> handle( pyObj );
                    py::numeric::array arr( handle );
                    return arr.copy();

            }
            py::object calc_modelerr(boost::function<void(const vtype&, vtype&, dtype)> ode, int ndim, py::numeric::array data, dtype dt) {
                    npy_intp size[1];
                    size[0] = len(data)-1;

                    dtype out[len(data)-1];
                    runge_kutta4< vtype > stepper;

                    vtype curr(ndim);
                    vtype nex(ndim);
                    vtype predict(ndim);

                    for(int i = 0; i < len(data)-1; i++)
                    {
                            for (int j = 0; j < ndim; j++)
                            {
                                    curr[j] = py::extract<dtype>(data[i][j]);
                                    nex[j] = py::extract<dtype>(data[i+1][j]);
                            }

                            stepper.do_step(ode,curr,i*dt,predict,dt);

                            out[i] = 0.0;
                            for (int j = 0; j < ndim; j++) {
                                    out[i] += predict[j] - nex[j];
                            }

                    }

                    PyObject * pyObj = PyArray_SimpleNewFromData( 1, size, NPY_DOUBLE, out );
                    py::handle<> handle( pyObj );
                    py::numeric::array arr( handle );
                    return arr.copy();

            }
};

class iz: public neuron
{
    public:
            dtype a,b,c,d,h;
            iz() {
                    a = 0.02;
                    b = 0.2;
                    c = -65.0;
                    d = 8.0;
                    h = 30.0;
            };
            iz(dtype na, dtype nb, dtype nc, dtype nd)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            h = 30.0;
                    };
            iz(dtype na, dtype nb, dtype nc, dtype nd, dtype nh)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            h = nh;
                    };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            dtype v = c;
                            dtype u = -14.0;

                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            dtype data[grid][2];

                            for(int i = 0; i < grid; i++)
                            {
                                    rt = round(i*dt)/res;

                                    v += dt*(0.04*v*v + 5*v + 140 - u + iapp[rt]);
                                    u += dt*(a*(b*v - u));

                                    if(v >= h)
                                    {
                                            data[i][0] = h;
                                            data[i][1] = u;
                                            v = c;
                                            u = u + d;
                                    }
                                    else {
                                            data[i][0] = v;
                                            data[i][1] = u;
                                    }
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return arr.copy();
                    }
};

class adex: public neuron
{
    public:
            dtype C,gl,el,delt,vt,tw,a,vr,b,h,R;
            adex() 
            {
                C    = 1.0; 
                gl   = 30.0;
                el   = -70.6;
                delt = 2.0;
                vt   = -55.0;
                tw   = 144.0;
                a    = 4.0;
                vr   = -70.6;
                b    = 80.5;
                h    = 30.0;
                R    = 1.0;
            };
            adex(dtype nC,dtype ngl,dtype nel,dtype ndelt,dtype nvt,dtype ntw,dtype na,dtype nvr,dtype nb)
            {
                C    = nC; 
                gl   = ngl;
                el   = nel;
                delt = ndelt;
                vt   = nvt;
                tw   = ntw;
                a    = na;
                vr   = nvr;
                b    = nb;
                h    = 30.0;
                R    = 1.0;
            };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            dtype v = el;
                            dtype w = 0.0;

                            dtype dv = 0.0;
                            dtype dw = 0.0;

                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            dtype data[grid][2];

                            for(int i = 0; i < grid; i++)
                            {
                                    rt = round(i*dt)/res;


                                    dv = dt*(1/C*(-gl*(v-el) + gl*delt*exp((v-vt)/delt) - w + R*iapp[rt]));
                                    dw = dt*(1/tw*(a*(v-el) - w));
                                    
                                    v += dv;
                                    w += dw;
                                    
                                    if(v >= h)
                                    {
                                            data[i][0] = h;
                                            data[i][1] = w;
                                            v = vr;
                                            w += b;
                                    }
                                    else {
                                            data[i][0] = v;
                                            data[i][1] = w;
                                    }
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return arr.copy();
                    }
};

class ad2ex: public neuron
{
    public:
            dtype C,gl,el,delt,vt0,tw,a,vr,b,tt,c,h,R;
            ad2ex() 
            {
                C    = 1.0; 
                gl   = 30.0;
                el   = -70.6;
                delt = 2.0;
                vt0  = -55.0;
                tw   = 144.0;
                a    = 4.0;
                vr   = -70.6;
                b    = 80.5;
                h    = 30.0;
                tt   = 300.0;
                c    = 20.0;
                R    = 1.0;
            };
            ad2ex(dtype nC,dtype ngl,dtype nel,dtype ndelt,dtype nvt0,dtype ntw,dtype na,dtype nvr,dtype nb,dtype ntt,dtype nc)
            {
                C    = nC; 
                gl   = ngl;
                el   = nel;
                delt = ndelt;
                vt0  = nvt0;
                tw   = ntw;
                a    = na;
                vr   = nvr;
                b    = nb;
                tt   = ntt;
                c    = nc;
                h    = 30.0;
                R    = 1.0;
            };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;

                            int rt = 0;

                            dtype v  = el;
                            dtype w  = 0.0;
                            dtype vt = vt0;
 
                            dtype dv  = 0.0;
                            dtype dw  = 0.0;
                            dtype dvt = 0.0;

                            npy_intp size[3];
                            size[0] = grid;
                            size[1] = 3;

                            dtype data[grid][3];

                            for(int i = 0; i < grid; i++)
                            {
                                    rt = round(i*dt)/res;
                                    dv = dt*(1/C*(-gl*(v-el) + gl*delt*exp((v-vt)/delt) - w + R*iapp[rt]));
                                    dw = dt*(1/tw*(a*(v-el) - w));
                                    dvt = (1/tt*(vt0 - vt));

                                    v  += dv;
                                    w  += dw;
                                    vt += dvt;
                                    
                                    if(v >= h)
                                    {
                                            data[i][0] = h;
                                            data[i][1] = w;
                                            data[i][2] = vt;

                                            v = vr;
                                            w += b;
                                            vt += c;
                                    }
                                    else {
                                            data[i][0] = v;
                                            data[i][1] = w;
                                            data[i][2] = vt;
                                    }
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return arr.copy();
                    }
};

class mat: public neuron
{
    public:
            dtype tm,R,a,b,w,t1,t2,h,tref;
            mat() 
            {
                tm=10;
                R=50;
                a=15;
                b=3;
                w=5;
                t1=10;
                t2=200;
                tref=2;
                            };
            mat(dtype na, dtype nb, dtype nw)
            {
                a  = na;
                b  = nb;
                w  = nw;
                tm  = 10;
                R  = 50;
                t1 = 10;
                t2 = 200;
                tref = 2;
            };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            dtype v = 0;
                            dtype h = 0;

                            dtype h1 = 0;
                            dtype h2 = 0;
                            dtype tf = 0;

                            dtype dv = 0;
                            dtype dh1 = 0;
                            dtype dh2 = 0;
                            
                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            dtype data[grid][2];

                            py::list spikes;

                            for(int i = 0; i < grid; i++)
                            {

                                rt = round(i*dt)/res;

                                dh1 = dt*(-h1/t1);
                                dh2 = dt*(-h2/t2);

                                dv  = dt*(-v + R*iapp[rt])/tm;

                                v  = v + dv;
                                h1 = h1 + dh1;
                                h2 = h2 + dh2;

                                h  = w + h1 + h2;

                                if(v > h && tf + tref/dt < i + 1)
                                {

                                    h1 += a;
                                    h2 += b;

                                    tf = i + 1;
                                    spikes.append(i);
                                }

                                data[i][0] = v;
                                data[i][1] = h;   
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return py::make_tuple(arr,spikes);
                    }
};

class augmat: public neuron
{
    public:
            dtype tm,R,a,b,w,t1,t2,h,tref,tv,c;
            augmat() 
            {
                tm=10;
                R=50;
                a=-0.5;
                b=0.35;
                w=5;
                t1=10;
                t2=200;
                tref=2;
                tv=5;
                c=0.3;
            };
            augmat(dtype na, dtype nb, dtype nc, dtype nw)
            {
                a  = na;
                b  = nb;
                w  = nw;
                c  = nc;
                tm  = 10;
                R  = 50;
                t1 = 10;
                t2 = 200;
                tref = 2;
                tv = 5;

            };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            dtype v = 0;
                            dtype h = 0;

                            dtype h1 = 0;
                            dtype h2 = 0;
                            dtype hv = 0;
                            dtype tf = 0;

                            dtype dv = 0;
                            dtype dh1 = 0;
                            dtype dh2 = 0;
                            dtype dhv = 0;

                            dtype ddhv = 0;
                            
                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            dtype data[grid][2];

                            py::list spikes;

                            for(int i = 0; i < grid; i++)
                            {

                                rt = round(i*dt)/res;

                                dh1 = dt*(-h1/t1);
                                dh2 = dt*(-h2/t2);
                                dhv = dt*(-hv/tv + ddhv);
                                ddhv = ddhv + dt*(-(hv/tv + dhv)/tv) + c*dv;

                                dv  = dt*(-v + R*iapp[rt])/tm;

                                v  = v + dv;
                                h1 = h1 + dh1;
                                h2 = h2 + dh2;
                                hv = hv + dhv;

                                h  = w + h1 + h2 + hv;

                                if(v > h && tf + tref/dt < i + 1)
                                {

                                    h1 += a;
                                    h2 += b;

                                    tf = i + 1;
                                    spikes.append(i);
                                }

                                data[i][0] = v;
                                data[i][1] = h;   
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return py::make_tuple(arr,spikes);
                    }
};

class hr: public neuron
{
    private:
            int ndim = 3;

    public:
            typedef boost::array< dtype , 3> state_type; // holds variable values form previous timestep
            dtype a,b,c,d,r,s,xn;
            hr() {
                    a = 1.0;
                    b = 3.0;
                    c = 1.0;
                    d = 5.0;
                    r = 0.005;
                    s = 4.0;
                    xn = -2.0;
            };

            hr(dtype na, dtype nb, dtype nc, dtype nd, dtype nr, dtype ns, dtype nxn)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            r = nr;
                            s = ns;
                            xn = nxn;
                    };

            boost::function<void(const vtype&, vtype&, dtype)> ode = [this]( const vtype &x , vtype &dxdt , dtype t )
                    {
                            // Getting the applied current at time t
                            int rt = round(t)/res;
                            dtype I = iapp[rt];

                            // Membrane potential
                            dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                            // Spiking variable/Recovery Current
                            dxdt[1] = c - d*x[0]*x[0] - x[1];

                            // Bursting variable
                            dxdt[2] = r*(s*(x[0] - xn) - x[2]);
                    };


            py::object simulate(dtype tspan, dtype dt){
                    vtype start = {-2.0, 0.0, 3.0};
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt){
                    return calc_modelerr(ode, ndim,data, dt);
            }
};

class hr4: public neuron
{
    private:
            int ndim = 4;

    public:
            typedef boost::array< dtype , 4> state_type; // holds variable values form previous timestep
            dtype a,b,c,d,r,s,xn,v,g,k,l;
            hr4() {
                    a = 1.0;
                    b = 3.0;
                    c = 1.0;
                    d = 5.0;
                    r = 0.005;
                    s = 4.0;
                    xn = -2.0;
                    v = 0.001;
                    g = 0.1;
                    k = 3.0;
                    l = 1.6;
            };

            hr4(dtype na, dtype nb, dtype nc, dtype nd, dtype nr, dtype ns, dtype nxn, dtype nv, dtype nk, dtype ng, dtype nl)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            r = nr;
                            s = ns;
                            xn = nxn;
                            v = nv;
                            g = ng;
                            k = nk;
                            l = nl;
                    };

            boost::function<void(const vtype&, vtype&, dtype)> ode = [this]( const vtype &x , vtype &dxdt , dtype t )
                    {
                            // Getting the applied current at time t
                            int rt = round(t)/res;
                            dtype I = iapp[rt];

                            // Membrane potential
                            dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                            // Spiking variable/Recovery Current
                            dxdt[1] = c - d*x[0]*x[0] - x[1] - g*x[3];

                            // Bursting variable
                            dxdt[2] = r*(s*(x[0] - xn) - x[2]);

                            dxdt[3] = v*(k*(x[1] + l) - x[3]);
                    };


            py::object simulate(dtype tspan, dtype dt){
                    vtype start = {-2.0, 0.0, 3.0, -2};
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt){
                    return calc_modelerr(ode, ndim,data, dt);
            }
};

class hr_alt: public neuron
{
    private:
            int ndim = 3;

    public:
            typedef boost::array< dtype , 3> state_type; // holds variable values form previous timestep
            dtype a,b,c,d,r,s,xn;
            hr_alt() {
                    a = 1.0;
                    b = 3.0;
                    c = 1.0;
                    d = 5.0;
                    r = 0.005;
                    s = 0.02;
                    xn = -2.0;
            };

            hr_alt(dtype na, dtype nb, dtype nc, dtype nd, dtype nr, dtype ns, dtype nxn)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            r = nr;
                            s = ns;
                            xn = nxn;
                    };

            boost::function<void(const vtype&, vtype&, dtype)> ode = [this]( const vtype &x , vtype &dxdt , dtype t )
                    {
                            // Getting the applied current at time t
                            int rt = round(t)/res;
                            dtype I = iapp[rt];

                            // Membrane potential
                            dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                            // Spiking variable/Recovery Current
                            dxdt[1] = c - d*x[0]*x[0] - x[1];

                            // Bursting variable
                            dxdt[2] = s*(x[0] - xn) - r*x[2];
                    };


            py::object simulate(dtype tspan, dtype dt){
                    vtype start = {-2.0, 0.0, 3.0};
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt){
                    return calc_modelerr(ode, ndim,data, dt);
            }
};

class hr2: public neuron
{
    private:
            int ndim = 2;

    public:
            typedef boost::array< dtype , 3> state_type; // holds variable values form previous timestep
            dtype a,b,c,d,xn;
            hr2() {
                    a = 1.0;
                    b = 3.0;
                    c = 1.0;
                    d = 5.0;
                    xn = -2.0;
            };

            hr2(dtype na, dtype nb, dtype nc, dtype nd, dtype nr, dtype ns, dtype nxn)
                    {
                            a = na;
                            b = nb;
                            c = nc;
                            d = nd;
                            xn = nxn;
                    };

            boost::function<void(const vtype&, vtype&, dtype)> ode = [this]( const vtype &x , vtype &dxdt , dtype t )
                    {
                            // Getting the applied current at time t
                            int rt = round(t)/res;
                            dtype I = iapp[rt];

                            // Membrane potential
                            dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                            // Spiking variable/Recovery Current
                            dxdt[1] = c - d*x[0]*x[0] - x[1];

                    };


            py::object simulate(dtype tspan, dtype dt){
                    vtype start = {-2.0, 0.0};
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt){
                    return calc_modelerr(ode, ndim,data, dt);
            }
};

class hh: public neuron
{
    private:
            int ndim = 4;
    public:
            dtype C,gna,Ena,gk,Ek,gl,El,vm,dvm,tm0,tm1,vmt,dvmt,vh,dvh,
                    th0,th1,vht,dvht,vn,dvn,tn0,tn1,vnt,dvnt;
            hh() {
                    C=1.0;
                    gna=120.0;
                    Ena=50.0;
                    gk=20.0;
                    Ek=-77.0;
                    gl=0.3;
                    El=-54.4;
                    vm=-40.0;
                    dvm=15.0;
                    tm0=0.1;
                    tm1=0.4;
                    vmt=-40.0;
                    dvmt=15.0;
                    vh=-60.0;
                    dvh=-15.0;
                    th0=1.0;
                    th1=7.0;
                    vht=-60.0;
                    dvht=-15.0;
                    vn=-55.0;
                    dvn=30.0;
                    tn0=1.0;
                    tn1=5.0;
                    vnt=-55.0;
                    dvnt=30.0;
                    res=0.1;
            };
            hh(py::list nparam)
                    {
                            C=1.0;
                            gna=py::extract<dtype>(nparam[0]);
                            Ena=py::extract<dtype>(nparam[1]);
                            gk=py::extract<dtype>(nparam[2]);
                            Ek=py::extract<dtype>(nparam[3]);
                            gl=py::extract<dtype>(nparam[4]);
                            El=py::extract<dtype>(nparam[5]);
                            vm=py::extract<dtype>(nparam[6]);
                            dvm=py::extract<dtype>(nparam[7]);
                            tm0=py::extract<dtype>(nparam[8]);
                            tm1=py::extract<dtype>(nparam[9]);
                            vmt=py::extract<dtype>(nparam[10]);
                            dvmt=py::extract<dtype>(nparam[11]);
                            vh=py::extract<dtype>(nparam[12]);
                            dvh=py::extract<dtype>(nparam[13]);
                            th0=py::extract<dtype>(nparam[14]);
                            th1=py::extract<dtype>(nparam[15]);
                            vht=py::extract<dtype>(nparam[16]);
                            dvht=py::extract<dtype>(nparam[17]);
                            vn=py::extract<dtype>(nparam[18]);
                            dvn=py::extract<dtype>(nparam[19]);
                            tn0=py::extract<dtype>(nparam[20]);
                            tn1=py::extract<dtype>(nparam[21]);
                            vnt=py::extract<dtype>(nparam[22]);
                            dvnt=py::extract<dtype>(nparam[23]);
                            res=0.1;
                    };
            boost::function<void(const vtype&, vtype&, dtype)> ode = [this]( const vtype &x , vtype &dxdt , dtype t )
                    {
                            int rt = round(round(t)/res);
                            dtype I = iapp[rt];

                            dxdt[0] = ((gna*x[1]*x[1]*x[1]*x[2]*(Ena-x[0]))+(gk*x[3]*x[3]*x[3]*x[3]*(Ek - x[0]))+
                                       (gl*(El-x[0])) + I)/C;

                            dtype taum = tm0 + tm1 * (1-pow(tanh((x[0] - vmt)/dvmt),2));
                            dtype m0 = (1+tanh((x[0]-vm)/dvm))/2;
                            dxdt[1] = (m0 - x[1])/taum;

                            dtype tauh = th0 + th1 * (1-pow(tanh((x[0] - vht)/dvht),2));
                            dtype h0 = (1+tanh((x[0]-vh)/dvh))/2;
                            dxdt[2] = (h0 - x[2])/tauh;

                            dtype taun = tn0 + tn1 * (1-pow(tanh((x[0] - vnt)/dvnt),2));
                            dtype n0 = (1+tanh((x[0]-vn)/dvn))/2;
                            dxdt[3] = (n0 - x[3])/taun;

                    };
            py::object simulate(dtype tspan, dtype dt){
                    vtype start = { -60.0, 0.0, 0.6, 0.5}; // initial
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt){
                    return calc_modelerr(ode, ndim,data, dt);
            }
};

class pasiv: public neuron
{
    private:
            int ndim = 2;
            dtype _ge, _gm, _Cm, _Ce, _Vr;
    public:
            pasiv()
                    : _ge(1000 / 10), _gm(1000 / 500), _Cm(50), _Ce(5), _Vr(-60) {}
            pasiv(dtype Cm, dtype Ce, dtype Rm, dtype Re, dtype Vr)
                    : _ge(1000 / Re), _gm(1000 / Rm), _Cm(Cm), _Ce(Ce), _Vr(Vr) {}

            boost::function<void(vtype const &, vtype &, dtype)> ode =
                    [this]( vtype const & x , vtype & dxdt , dtype t ) {
                    int i = round(t / res);
                    dtype I = iapp[i];
                    dtype Vin = x[0];
                    dtype Vm  = x[1];
                    dxdt[0] = (I - (Vin - Vm) * _ge) / _Ce;
                    dxdt[1] = ((Vin - Vm) * _ge - (Vm - _Vr) * _gm) / _Cm;
            };
            py::object simulate(dtype tspan, dtype dt, dtype init) {
                    vtype start = {init, init}; // initial
                    return integrate(ode, ndim, tspan, dt, start);
            }

            py::object modelerr(py::numeric::array data, dtype dt) {
                    return calc_modelerr(ode, ndim, data, dt);
            }
};

class matex: public neuron
{
    public:
            dtype C,gl,el,delt,vt0,tw,a,vr,b,h,R,l,m,n;
            matex() 
            {
                C    = 1.0; 
                gl   = 30.0;
                el   = -70.6;
                delt = 2.0;
                vt0  = -55.0;
                tw   = 144.0;
                a    = 4.0;
                vr   = -70.6;
                b    = 80.5;
                h    = 30.0;
                R    = 1.0;
                l    = 15;
                m    = 3;
                n    = 0.3;
            };
            matex(dtype nC,dtype ngl,dtype nel,dtype ndelt,dtype nvt0,dtype ntw,dtype na,dtype nvr,dtype nb, dtype nl, dtype nm, dtype nn)
            {
                C    = nC; 
                gl   = ngl;
                el   = nel;
                delt = ndelt;
                vt0  = nvt0;
                tw   = ntw;
                a    = na;
                vr   = nvr;
                b    = nb;
                h    = 30.0;
                R    = 1.0;
                l    = nl;
                m    = nm;
                n    = nn;
            };

            py::object simulate(const dtype duration, const dtype dt)
                    {
                            int grid = duration/dt;

                            int rt = 0;

                            dtype v  = el;
                            dtype w  = 0.0;
                            dtype vt = vt0;
 

                            dtype vt1 = 0;
                            dtype vt2 = 0;
                            dtype vt3 = 0;

                            dtype dvt1 = 0;
                            dtype dvt2 = 0;
                            dtype dvt3 = 0;
                            dtype ddvt3 = 0;

                            dtype dv  = 0.0;
                            dtype dw  = 0.0;

                            npy_intp size[3];
                            size[0] = grid;
                            size[1] = 3;

                            dtype data[grid][3];

                            for(int i = 0; i < grid; i++)
                            {
                                    rt = round(i*dt)/res;

                                    dvt1 = dt*(-vt1/10);
                                    dvt2 = dt*(-vt2/200);
                                    dvt3 = dt*(-vt3/5 + ddvt3);
                                    ddvt3 = ddvt3 + dt*(-(vt3/5 + dvt3)/5) + n*dv;

                                    dv = dt*(1/C*(-gl*(v-el) + gl*delt*exp((v-vt)/delt) - w + R*iapp[rt]));
                                    dw = dt*(1/tw*(a*(v-el) - w));


                                    vt1 = vt1 + dvt1;
                                    vt2 = vt2 + dvt2;
                                    vt3 = vt3 + dvt3;


                                    v  += dv;
                                    w  += dw;
                                    vt = vt0 + vt1 + vt2 + vt3;
                                   
                                    if(v >= h)
                                    {
                                            data[i][0] = h;
                                            data[i][1] = w;
                                            data[i][2] = vt;

                                            dv = dt*(h + vr);

                                            v = vr;
                                            w += b;
                                            
                                            vt1 += l;
                                            vt2 += m;
                                    }
                                    else {
                                            data[i][0] = v;
                                            data[i][1] = w;
                                            data[i][2] = vt;
                                    }
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return arr.copy();
                    }
};

BOOST_PYTHON_MODULE(cneurons)
{
        using namespace py;
        import_array();
        py::numeric::array::set_module_and_type("numpy", "ndarray");

        class_<neuron, boost::noncopyable>("neuron")
                .def("apply_current", &neuron::apply_current);

        class_<iz,bases<neuron>>("iz")
                .def(init<dtype, dtype, dtype, dtype>())
                .def("simulate", &iz::simulate)
                .def_readwrite("a", &iz::a)
                .def_readwrite("b", &iz::b)
                .def_readwrite("c", &iz::c)
                .def_readwrite("d", &iz::d)
                .def_readwrite("h", &iz::h);

        
        class_<adex,bases<neuron>>("adex")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &adex::simulate)
                .def_readwrite("C", &adex::C)
                .def_readwrite("gl", &adex::gl)
                .def_readwrite("el", &adex::el)
                .def_readwrite("delt", &adex::delt)
                .def_readwrite("vt", &adex::vt)
                .def_readwrite("tw", &adex::tw)
                .def_readwrite("a", &adex::a)
                .def_readwrite("vr", &adex::vr)
                .def_readwrite("b", &adex::b)
                .def_readwrite("h", &adex::h)
                .def_readwrite("R", &adex::R);

        class_<ad2ex,bases<neuron>>("ad2ex")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &ad2ex::simulate)
                .def_readwrite("C", &ad2ex::C)
                .def_readwrite("gl", &ad2ex::gl)
                .def_readwrite("el", &ad2ex::el)
                .def_readwrite("delt", &ad2ex::delt)
                .def_readwrite("vt0", &ad2ex::vt0)
                .def_readwrite("tw", &ad2ex::tw)
                .def_readwrite("a", &ad2ex::a)
                .def_readwrite("vr", &ad2ex::vr)
                .def_readwrite("b", &ad2ex::b)
                .def_readwrite("h", &ad2ex::h)
                .def_readwrite("tt", &ad2ex::tt)
                .def_readwrite("c", &ad2ex::c)
                .def_readwrite("R", &ad2ex::R);



        class_<matex,bases<neuron>>("matex")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &matex::simulate)
                .def_readwrite("C", &matex::C)
                .def_readwrite("gl", &matex::gl)
                .def_readwrite("el", &matex::el)
                .def_readwrite("delt", &matex::delt)
                .def_readwrite("vt0", &matex::vt0)
                .def_readwrite("tw", &matex::tw)
                .def_readwrite("a", &matex::a)
                .def_readwrite("vr", &matex::vr)
                .def_readwrite("b", &matex::b)
                .def_readwrite("h", &matex::h)
                .def_readwrite("l", &matex::l)
                .def_readwrite("m", &matex::m)
                .def_readwrite("n", &matex::n)
                .def_readwrite("R", &matex::R);

        class_<mat,bases<neuron>>("mat")
                .def(init<dtype,dtype,dtype>())
                .def("simulate", &mat::simulate)
                .def_readwrite("tm", &mat::tm)
                .def_readwrite("R", &mat::R)
                .def_readwrite("a", &mat::a)
                .def_readwrite("b", &mat::b)
                .def_readwrite("w", &mat::w)
                .def_readwrite("t1", &mat::t1)
                .def_readwrite("t2", &mat::t2)
                .def_readwrite("tref", &mat::tref);

        class_<augmat,bases<neuron>>("augmat")
                .def(init<dtype,dtype,dtype,dtype>())
                .def("simulate", &augmat::simulate)
                .def_readwrite("tm", &augmat::tm)
                .def_readwrite("R", &augmat::R)
                .def_readwrite("a", &augmat::a)
                .def_readwrite("b", &augmat::b)
                .def_readwrite("w", &augmat::w)
                .def_readwrite("t1", &augmat::t1)
                .def_readwrite("t2", &augmat::t2)
                .def_readwrite("tref", &augmat::tref)
                .def_readwrite("tv", &augmat::tv)
                .def_readwrite("c", &augmat::c);


        class_<hr,bases<neuron>>("hr")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &hr::simulate)
                .def("modelerr", &hr::modelerr)
                .def_readwrite("a", &hr::a)
                .def_readwrite("b", &hr::b)
                .def_readwrite("c", &hr::c)
                .def_readwrite("d", &hr::d)
                .def_readwrite("r", &hr::r)
                .def_readwrite("s", &hr::s)
                .def_readwrite("xn", &hr::xn);

        class_<hr_alt,bases<neuron>>("hr_alt")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &hr_alt::simulate)
                .def("modelerr", &hr_alt::modelerr)
                .def_readwrite("a", &hr_alt::a)
                .def_readwrite("b", &hr_alt::b)
                .def_readwrite("c", &hr_alt::c)
                .def_readwrite("d", &hr_alt::d)
                .def_readwrite("r", &hr_alt::r)
                .def_readwrite("s", &hr_alt::s)
                .def_readwrite("xn", &hr_alt::xn);

        class_<hr2,bases<neuron>>("hr2")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &hr2::simulate)
                .def("modelerr", &hr2::modelerr)
                .def_readwrite("a", &hr2::a)
                .def_readwrite("b", &hr2::b)
                .def_readwrite("c", &hr2::c)
                .def_readwrite("d", &hr2::d)
                .def_readwrite("xn", &hr2::xn);

        class_<hr4,bases<neuron>>("hr4")
                .def(init<dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype>())
                .def("simulate", &hr4::simulate)
                .def("modelerr", &hr4::modelerr)
                .def_readwrite("a", &hr4::a)
                .def_readwrite("b", &hr4::b)
                .def_readwrite("c", &hr4::c)
                .def_readwrite("d", &hr4::d)
                .def_readwrite("r", &hr4::r)
                .def_readwrite("s", &hr4::s)
                .def_readwrite("xn",&hr4::xn)
                .def_readwrite("v", &hr4::v)
                .def_readwrite("g", &hr4::g)
                .def_readwrite("k", &hr4::k)
                .def_readwrite("l", &hr4::l);

        class_<hh,bases<neuron>>("hh")
                .def(init<py::list>())
                .def("simulate", &hh::simulate)
                .def("modelerr", &hh::modelerr)
                .def_readwrite("C", &hh::C)
                .def_readwrite("gna", &hh::gna)
                .def_readwrite("Ena", &hh::Ena)
                .def_readwrite("gk", &hh::gk)
                .def_readwrite("Ek", &hh::Ek)
                .def_readwrite("gl", &hh::gl)
                .def_readwrite("El", &hh::El)
                .def_readwrite("vm", &hh::vm)
                .def_readwrite("dvm", &hh::dvm)
                .def_readwrite("tm0", &hh::tm0)
                .def_readwrite("tm1", &hh::tm1)
                .def_readwrite("vmt", &hh::vmt)
                .def_readwrite("dvmt", &hh::dvmt)
                .def_readwrite("vh", &hh::vh)
                .def_readwrite("dvh", &hh::dvh)
                .def_readwrite("th0", &hh::th0)
                .def_readwrite("th1", &hh::th1)
                .def_readwrite("vht", &hh::vht)
                .def_readwrite("dvht", &hh::dvht)
                .def_readwrite("vn", &hh::vn)
                .def_readwrite("dvn", &hh::dvn)
                .def_readwrite("tn0", &hh::tn0)
                .def_readwrite("tn1", &hh::tn1)
                .def_readwrite("vnt", &hh::vnt)
                .def_readwrite("dvnt", &hh::dvnt);

        class_<pasiv,bases<neuron>>("pasiv")
                .def(init<dtype,dtype,dtype,dtype,dtype>())
                .def("simulate", &pasiv::simulate);
}
