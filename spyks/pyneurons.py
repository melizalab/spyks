from __future__ import division
import numpy as np
import math
from scipy.integrate import odeint

class models:

  class hr:
      def __init__(self, name = 'Hindmarsh-Rose Neuron', a = 1, b = 3, c = 1, 
                   d = 5, r = 0.005, s = 4, x0 = -2,):
         self.name = name
         
         # Fast ion channel parameters
         self.a  = a
         self.b  = b
         self.c  = c
         self.d  = d
         
         # Slow ion channel parameter
         self.r  = r
          
         # Accommodation/Adaptation parameter
         self.s  = s
         
         # Resting potential
         self.x0 = x0
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, init, t):
         (x, y, z) = init
         
         rt = round(t) if round(t) < self.Tspan else self.Tspan -1
         rt = rt/self.Iapp_res
          
         # Membrane potential
         dx_dt = y - self.a*x**3+ self.b*x**2 - z + self.Iapp[rt]
         
         # Spiking variable/Recovery Current
         dy_dt = self.c - self.d*x**2 - y
         
         # Bursting variable
         dz_dt = self.r * (self.s * (x - self.x0) - z)
         return [dx_dt, dy_dt, dz_dt]
      
      def simulate(self, duration, dt=0.1, init=[-2,0,3]):
          self.tgrid = np.arange(0, duration, dt)
          out = odeint(self.ode, init, self.tgrid)
          return out

  class hr_alt:
      def __init__(self, name = 'Hindmarsh-Rose Neuron', a = 1, b = 3, c = 1, 
                   d = 5, r = 0.005, s = 0.02, x0 = -2,):
         self.name = name
         
         # Fast ion channel parameters
         self.a  = a
         self.b  = b
         self.c  = c
         self.d  = d
         
         # Slow ion channel parameter
         self.r  = r
          
         # Accommodation/Adaptation parameter
         self.s  = s
         
         # Resting potential
         self.x0 = x0
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, init, t):
         (x, y, z) = init
         
         rt = round(t) if round(t) < self.Tspan else self.Tspan -1
         rt = rt/self.Iapp_res
          
         # Membrane potential
         dx_dt = y - self.a*x**3+ self.b*x**2 - z + self.Iapp[rt]
         
         # Spiking variable/Recovery Current
         dy_dt = self.c - self.d*x**2 - y
         
         # Bursting variable
         dz_dt = (self.s * (x - self.x0) - self.r*z)

         return [dx_dt, dy_dt, dz_dt]
      
      def simulate(self, duration, dt=0.1, init=[-2,0,3]):
          self.tgrid = np.arange(0, duration, dt)
          out = odeint(self.ode, init, self.tgrid)
          return out

  class hr4:
      def __init__(self, name = 'Hindmarsh-Rose Neuron', a = 1, b = 3, c = 1, 
                   d = 5, r = 0.005, s = 4, x0 = -2, v=0.001,k=1.0,g=0.1):
         self.name = name
         
         # Fast ion channel parameters
         self.a  = a
         self.b  = b
         self.c  = c
         self.d  = d
         
         # Slow ion channel parameter
         self.r  = r
          
         # Accommodation/Adaptation parameter
         self.s  = s
         
         # Resting potential
         self.x0 = x0

         # add-on variables
         self.v = v
         self.k = k
         self.g = g
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, init, t):
         (x, y, z, w) = init
         
         rt = round(t) if round(t) < self.Tspan else self.Tspan -1
         rt = rt/self.Iapp_res
          
         # Membrane potential
         dx_dt = y - self.a*x**3+ self.b*x**2 - z + self.Iapp[rt]
         
         # Spiking variable/Recovery Current
         dy_dt = self.c - self.d*x**2 - y - self.g*w
         
         # Bursting variable
         dz_dt = self.r * (self.s * (x - self.x0) - z)

         dw_dt = self.v * (self.k * (y + 1) - w)

         return [dx_dt, dy_dt, dz_dt, dw_dt]
      
      def simulate(self, duration, dt=0.1, init=[-2,0,3,-2]):
          self.tgrid = np.arange(0, duration, dt)
          out = odeint(self.ode, init, self.tgrid)
          return out

  class hh:
      def __init__(self, name='Hodgkin-Huxley Neuron',C=1.0, gna=120.0, 
                   Ena=50.0, gk=20, Ek=-77.0, gl=0.3, El=-54.4, vm=-40.0, 
                   dvm=15.0, tm0=0.1, tm1=0.4, vmt=-40.0, dvmt=15.0, 
                   vh=-60.0, dvh=-15.0, th0=1.0, th1=7.0, vht=-60, 
                   dvht=-15.0, vn=-55.0, dvn=30.0, tn0=1.0, tn1=5.0, 
                   vnt=-55.0, dvnt=30.0):
          
          self.name = name 
                
          self.C    = C    # uF/cm^2
          self.gna  = gna  # mS/cm^2   # Na+ maximal conductance
          self.Ena  = Ena  # mV        # Na+ nernst potential 
          self.gk   = gk   # mS/cm^2   # K+ maximal conductance 
          self.Ek   = Ek   # mV        # K+ nernst potential 
          self.gl   = gl   # mS/cm^2   # leak maximal conductance
          self.El   = El   # mV        # leak nernst potential

          # Na+ open gating variables
          self.vm   = vm   # mV        
          self.dvm  = dvm  # mV
          self.tm0  = tm0  # ms
          self.tm1  = tm1  # ms
          self.vmt  = vmt  # mV
          self.dvmt = dvmt # mV

          # Na+ closed gating variables
          self.vh   = vh   # mV
          self.dvh  = dvh  # mV
          self.th0  = th0  # ms
          self.th1  = th1  # ms
          self.vht  = vht   # mV
          self.dvht = dvht # mV

          # K open gating variables
          self.vn   = vn   # mV
          self.dvn  = dvn  # mV
          self.tn0  = tn0  # ms
          self.tn1  = tn1  # ms
          self.vnt  = vnt  # mV
          self.dvnt = dvnt # mV
      
      def apply_current(self, current, res=1):
          self.Iapp = current
          self.Iapp_res = res
          self.Tspan = len(self.Iapp)*res
          
      def ode(self, init, t):
          V, m, h, n = init

          rt = round(t) if round(t) < self.Tspan else self.Tspan - 1
          rt = rt/self.Iapp_res
          
          dV_dt = ((self.gna*math.pow(m,3)*h*(self.Ena-V))+(self.gk*math.pow(n,4)*(self.Ek - V))+
                   (self.gl*(self.El-V)) + self.Iapp[rt])/self.C

          taum = self.tm0 + self.tm1 * (1-math.pow(math.tanh((V - self.vmt)/self.dvmt),2))
          m0 = (1+math.tanh((V-self.vm)/self.dvm))/2
          dm_dt = (m0 - m)/taum

          tauh = self.th0 + self.th1 * (1-math.pow(math.tanh((V - self.vht)/self.dvht),2))
          h0 = (1+math.tanh((V-self.vh)/self.dvh))/2
          dh_dt = (h0 - h)/tauh

          taun = self.tn0 + self.tn1 * (1-math.pow(math.tanh((V - self.vnt)/self.dvnt),2))
          n0 = (1+math.tanh((V-self.vn)/self.dvn))/2
          dn_dt = (n0 - n)/taun

          return [dV_dt, dm_dt, dh_dt, dn_dt]
      
      def simulate(self, duration, dt=0.1, init=[-60,0.0,0.6,0.5]):
          self.tgrid = np.arange(0, duration, dt)
          out = odeint(self.ode, init, self.tgrid)
          return out

  class iz:
      def __init__(self, name = 'Izhikevich Neuron', a = 0.02, b = 0.2, c = -65, 
                   d = 8, h = 30):
         self.name = name
         
         # Fast ion channel parameters
         self.a  = a
         self.b  = b
         self.c  = c
         self.d  = d
         self.h  = h
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, u) = init
         trace = np.zeros((len(self.tgrid),2))
         
         for i,j in enumerate(self.tgrid):
    	       dv = dt*(0.04*v*v + 5*v + 140 - u + self.Iapp[i])
  	       du = dt*(self.a*(self.b*v - u))

  	       v += dv
  	       u += du

  	       if v >= self.h:
  	       	  trace[i,0] = self.h
  	          v = self.c
  	          u = u + self.d

  	       else:
  	       	   trace[i,0] = v
  	       	   trace[i,1] = u

         return trace
      
      def simulate(self, duration, dt=0.1, init=[-60,0]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out

  class adex:
      def __init__(self, name = 'AdEx (aEIF) neuron', C = 281.0, gl = 30.0, el = -70.6, delt = 2, vt = -50.4,
                   tw = 144.0, a = 4, vr=-70.6, b = 80.5, h = 30):
         self.name = name
         
         # Fast ion channel parameters
         self.C  = C
         self.gl = gl
         self.el = el
         self.delt = delt
         self.vt = vt
         self.tw = tw
         self.a = a
         self.vr = vr
         self.b = b
         self.h = h
      
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, w) = init
         trace = np.zeros((len(self.tgrid),2))
         
         for i,j in enumerate(self.tgrid):
            dv = dt*(1/self.C*(-self.gl*(v-self.el) + self.gl*self.delt*np.exp((v-self.vt)/self.delt) - w + self.Iapp[i]))
            dw = dt*(1/self.tw*(self.a*(v-self.el) - w))

            v += dv
            w += dw

            if v >= self.h:
              trace[i,0] = self.h
              trace[i,1] = w
              v = self.vr
              w += self.b

            else:
              trace[i,0] = v
              trace[i,1] = w

         return trace
      
      def simulate(self, duration, dt=0.1, init=[-70,0]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out

  class ad2ex:
      def __init__(self, name = 'Ad2Ex (a2EIF) neuron', C = 281.0, gl = 30.0, el = -70.6, delt = 2, vt0 = -50.4,
                   tw = 144.0, a = 4, vr=-70.6, b = 80.5, h = 30, tt = 300.0, c=20.0):
         self.name = name
         
         # Fast ion channel parameters
         self.C  = C
         self.gl = gl
         self.el = el
         self.delt = delt
         self.vt0 = vt0
         self.tw = tw
         self.a = a
         self.vr = vr
         self.b = b
         self.h = h
         self.tt = tt
         self.c = c
      
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, w, vt) = init
         trace = np.zeros((len(self.tgrid),3))
         
         for i,j in enumerate(self.tgrid):
            dv = dt*(1/self.C*(-self.gl*(v-self.el) + self.gl*self.delt*np.exp((v-vt)/self.delt) - w + self.Iapp[i]))
            dw = dt*(1/self.tw*(self.a*(v-self.el) - w))

            dvt = (1/self.tt*(self.vt0 - vt))

            v += dv
            w += dw
            vt += dvt

            if v >= self.h:
              trace[i,0] = self.h
              trace[i,1] = w
              trace[i,2] = vt

              v   = self.vr
              w  += self.b
              vt += self.c

            else:
              trace[i,0] = v
              trace[i,1] = w
              trace[i,2] = vt

         return trace
      
      def simulate(self, duration, dt=0.1, init=[-70,0,-50]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out

  class mat:
      def __init__(self, name = 'Augmented MAT neuron',tm=10,R=50,a=15,b=3,w=5,t1=10,t2=200,tref=2):
         self.name = name
         
         # Fast ion channel parameters
         self.tm = tm
         self.R  = R
         self.a  = a
         self.b  = b
         self.w  = w
         self.t1 = t1
         self.t2 = t2

         self.tref = tref

      
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, h) = init
         trace = np.zeros((len(self.tgrid),4))
         spikes = []

         h1 = h2 = 0

         tf = 0
         
         for i,j in enumerate(self.tgrid):

            dh1 = dt*(-h1/self.t1)
            dh2 = dt*(-h2/self.t2)

            dv  = dt*(-v + self.R*self.Iapp[i])/self.tm


            v  = v + dv
            h1 = h1 + dh1
            h2 = h2 + dh2

            h  = self.w + h1 + h2

            if v > h and tf + self.tref/dt < i + 1:
              
              h1 += self.a
              h2 += self.b

              tf = i + 1
              spikes.append(i)

            trace[i,0] = v
            trace[i,1] = h
            
         return trace,spikes
      
      def simulate(self, duration, dt=0.1, init=[0,0]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out

  class augmat:
      def __init__(self, name = 'Augmented MAT neuron',tm=10,R=50,a=-0.5,b=0.35,w=5,c=-0.3,t1=10,t2=200,tv=5,tref=2):
         self.name = name
         
         # Fast ion channel parameters
         self.tm = tm
         self.R  = R
         self.a  = a
         self.b  = b
         self.w  = w
         self.c  = c
         self.t1 = t1
         self.t2 = t2
         self.tv = tv

         self.tref = tref

      
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, h) = init
         trace = np.zeros((len(self.tgrid),2))
         spikes = []

         h1 = h2 = hv = dhv = dv = ddhv = 0

         tf = 0
         
         for i,j in enumerate(self.tgrid):

            dh1 = dt*(-h1/self.t1)
            dh2 = dt*(-h2/self.t2)
            dhv = dt*(-hv/self.tv + ddhv) 
            ddhv = ddhv + dt*(-(hv/self.tv + dhv)/self.tv) + self.c*dv

            dv  = dt*(-v + self.R*self.Iapp[i])/self.tm


            v  = v  + dv
            h1 = h1 + dh1
            h2 = h2 + dh2
            hv = hv + dhv

            h  = self.w + h1 + h2 + hv

            if v > h and tf + self.tref/dt < i + 1:
              
              h1 += self.a
              h2 += self.b

              tf = i + 1
              spikes.append(i)

            trace[i,0] = v
            trace[i,1] = h
            
         return trace,spikes
      
      def simulate(self, duration, dt=0.1, init=[0,0]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out

  class matex:
      def __init__(self, name = 'MATEx (MAT + a2EIF) neuron', C = 281.0, gl = 30.0, el = -70.6, delt = 2, vt0 = -50.4,
                   tw = 144.0, a = 4, vr=-70.6, b = 80.5, h = 30, l=15, m=3):
         self.name = name
         
         # Fast ion channel parameters
         self.C  = C
         self.gl = gl
         self.el = el
         self.delt = delt
         self.vt0 = vt0
         self.tw = tw
         self.a = a
         self.vr = vr
         self.b = b
         self.h = h
         self.l = l
         self.m = m
      
          
      def apply_current(self, current, res=1):
         self.Iapp = current
         self.Iapp_res = res
         self.Tspan = len(self.Iapp)*res
      
      def ode(self, dt, init):
         (v, w, vt) = init
         trace = np.zeros((len(self.tgrid),3))

         vt1 = vt2 = dvt1 = dvt2 = 0
         
         for i,j in enumerate(self.tgrid):
         	dv = dt*(1/self.C*(-self.gl*(v-self.el) + self.gl*self.delt*np.exp((v-vt)/self.delt) - w + self.Iapp[i]))
         	dw = dt*(1/self.tw*(self.a*(v-self.el) - w))


         	dvt1 = dt*(-vt1/10)
         	dvt2 = dt*(-vt2/200)

         	vt1 = vt1 + dvt1
         	vt2 = vt2 + dvt2

         	v += dv
         	w += dw
         	vt = self.vt0 + vt1 + vt2

         	if v >= self.h:
         		trace[i,0] = self.h
         		trace[i,1] = w
         		trace[i,2] = vt

         		v   = self.vr
         		w  += self.b

         		vt1 += self.l
         		vt2 += self.m

         	else:
         		trace[i,0] = v
         		trace[i,1] = w
         		trace[i,2] = vt

         return trace
      
      def simulate(self, duration, dt=0.1, init=[-70,0,-50]):
          self.tgrid = np.arange(0, duration, dt)

          out = self.ode(dt,init)
          return out