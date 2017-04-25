# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Low-level routines: distance matrix generation, problem matrix generation, solver.
#
# JJ 2016-11-30

# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

from __future__ import absolute_import

from cython cimport view

# See the infrasructure module for the definition of Case.
cimport wlsqm.fitter.infra as infra

####################################################
# Distance matrix (c) generation
####################################################

cdef void make_c_nD( infra.Case* case, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D ) nogil
cdef void make_c_3D( infra.Case* case, double[::view.generic,::view.contiguous] xk ) nogil
cdef void make_c_2D( infra.Case* case, double[::view.generic,::view.contiguous] xk ) nogil
cdef void make_c_1D( infra.Case* case, double[::view.generic] xk ) nogil

####################################################
# Problem matrix (A) generation
####################################################

cdef void make_A( infra.Case* case ) nogil
cdef void preprocess_A( infra.Case* case, int debug ) nogil

####################################################
# RHS handling and solving
####################################################

cdef void solve( infra.Case* case, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid ) nogil
cdef int solve_iterative( infra.Case* case, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid, int max_iter,
                          double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D ) nogil

