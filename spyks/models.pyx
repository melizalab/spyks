# -*- coding: utf-8 -*-

from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "neurons.hpp" namespace "neurons":
    cdef cppclass adex:
        size_t N_PARAM
        size_t N_STATE
        size_t N_FORCING
        adex() except +
        void set_params(vector[double] & params)
        void set_forcing(vector[double] & forcing, double dt)
        bint reset(vector[double] & X)
        void operator()(vector[double] & X, vector[double] & dXdt, double time)

cdef class AdEx:
    cdef adex model

    @property
    def param_names(self):
        return ("C", "gl", "el", "delt", "vt", "tw", "a", "vr", "b", "h", "R")

    @property
    def forcing_names(self):
        return ("Iinj",)

    def set_params(self, params):
        self.model.set_params(params)

    def set_forcing(self, forcing, double dt):
        self.model.set_forcing(forcing, dt)

    def __call__(self, state, time):
        cdef vector[double] out
        out.resize(len(state))
        self.model(state, out, time)
        return out

    def reset(self, vector[double] state):
        cdef bint out
        out = self.model.reset(state)
        return state, out


# cdef class PyAdEx:
#     cdef size_t PARAM_DIM = 7
#     cdef double[:] params
#     cdef double[:] Iinj
#     cdef double dt

#     @property
#     def param_names(self):
#         return ("C", "gl", "el", "delt", "vt", "tw", "a", "vr", "b", "h", "R")

#     @property
#     def forcing_names(self):
#         return ("Iinj",)

#     def set_params(self, double[PARAM_DIM] params):
#         self.params = params.copy()

#     def set_forcing(self, double[:] Iinj, double dt):
#         self.Iinj = Iinj.copy()
#         self.dt = dt

#     def dxdt(self, X, t):
#          cdef double C, gl, el, delt, vt, tw, a, vr, b, h, R
#          C, gl, el, delt, vt, tw, a, vr, b, h, R = self.params
