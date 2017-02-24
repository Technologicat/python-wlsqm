# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Interpolation of fitted surrogate model.
#
# C API definitions.
#
# JJ 2016-11-30

from cython cimport view

cimport wlsqm.fitter.infra as infra

cdef int interpolate_nD( infra.Case* case, double[::view.generic,::view.contiguous] xManyD, double[::view.generic] x1D, double* out, int diff ) nogil

cdef int interpolate_3D( infra.Case* case, double[::view.generic,::view.contiguous] x, double* out, int diff ) nogil
cdef int interpolate_2D( infra.Case* case, double[::view.generic,::view.contiguous] x, double* out, int diff ) nogil
cdef int interpolate_1D( infra.Case* case, double[::view.generic] x, double* out, int diff ) nogil

