# -*- coding: utf-8 -*-
# -*- mode: cython -*-

cdef extern from "neurons.hpp" namespace "neurons":
    cdef cppclass adex:
        adex() except +
        void set_params
