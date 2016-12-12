# -*- coding: utf-8 -*-

from cython cimport view
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "neurons.hpp" namespace "neurons":
    cdef cppclass adex:
        size_t N_PARAM
        size_t N_STATE
        size_t N_FORCING
        adex() except +
        void set_params(double * params)
        void set_forcing(vector[double] & forcing, double dt)
        bint reset(double * X)
        void operator()(double * X, double * dXdt, double time)

cdef class AdEx:
    cdef adex model

    @property
    def param_names(self):
        return ("C", "gl", "el", "delt", "vt", "tw", "a", "vr", "b", "h", "R")

    @property
    def forcing_names(self):
        return ("Iinj",)

    def set_params(self, double[::1] params):
        #if (params.size != self.model.N_PARAM): raise ValueError
        self.model.set_params(&params[0])

    def set_forcing(self, forcing, double dt):
        self.model.set_forcing(forcing, dt)

    def __call__(self, double[::1] state, double time):
        cdef view.array out = view.array(shape=(self.model.N_STATE,), itemsize=sizeof(double), format="d")
        cdef double[:] dXdt = out
        self.model(&state[0], &dXdt[0], time)
        return out

    def reset(self, double[::1] state):
        cdef bint out
        out = self.model.reset(&state[0])
        return state, out


cdef class PyAdEx:
    cdef double C, gl, el, delt, vt, tw, a, vr, b, h, R
    cdef double[:] Iinj
    cdef double dt

    @property
    def param_names(self):
        return ("C", "gl", "el", "delt", "vt", "tw", "a", "vr", "b", "h", "R")

    @property
    def forcing_names(self):
        return ("Iinj",)

    def set_params(self, double[:] params):
        self.C, self.gl, self.el, self.delt, self.vt, self.tw, self.a, self.vr, self.b, self.h, self.R = params
        #self.params = params.copy()

    def set_forcing(self, double[:] Iinj, double dt):
        self.Iinj = Iinj.copy()
        self.dt = dt

    def dxdt(self, X, t):
        pass
         # cdef double C, gl, el, delt, vt, tw, a, vr, b, h, R
         # C, gl, el, delt, vt, tw, a, vr, b, h, R = self.params
