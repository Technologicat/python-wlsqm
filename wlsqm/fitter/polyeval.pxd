# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Evaluation of Taylor expansions and general polynomials up to 4th order in 1D, 2D and 3D.
#
# C API definitions.
#
# JJ 2016-12-09

from cython cimport view

cdef int taylor_3D( int order, double* fi, double xi, double yi, double zi, double[::view.generic,::view.contiguous] x, double* out ) nogil
cdef int general_3D( int order, double* fi, double xi, double yi, double zi, double[::view.generic,::view.contiguous] x, double* out ) nogil

cdef int taylor_2D( int order, double* fi, double xi, double yi, double[::view.generic,::view.contiguous] x, double* out ) nogil
cdef int general_2D( int order, double* fi, double xi, double yi, double[::view.generic,::view.contiguous] x, double* out ) nogil

cdef int taylor_1D( int order, double* fi, double xi, double[::view.generic] x, double* out ) nogil
cdef int general_1D( int order, double* fi, double xi, double[::view.generic] x, double* out ) nogil

