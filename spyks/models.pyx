# -*- coding: utf-8 -*-

from cython cimport view
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "neurons.hpp" namespace "neurons" nogil:
    cdef cppclass adex:
        size_t N_PARAM
        size_t N_STATE
        size_t N_FORCING
        adex() except +
        void set_params(double * params)
        void set_forcing(vector[double] & forcing, double dt)
        bint reset(vector[double] X)
        void operator()(vector[double] X, vector[double] dXdt, double time)

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

    # These methods are primarily for testing
    def __call__(self, vector[double] state, double time):
        #cdef view.array out = view.array(shape=(self.model.N_STATE,), itemsize=sizeof(double), format="d")
        cdef vector[double] dXdt = state.copy()
        self.model(state, dXdt, time)
        return dXdt

    def reset(self, vector[double] state):
        cdef bint out
        out = self.model.reset(state)
        return state, out
