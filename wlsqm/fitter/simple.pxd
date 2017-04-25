# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Cython declarations for the main module. See the .pyx source for wlsqm.fitter.simple for documentation.
#
# JJ 2016-11-07

# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True

# This module contains "driver" routines in the LAPACK sense.
# The low-level C routines are contained in wlsqm.fitter.impl.

from __future__ import absolute_import

from cython cimport view  # for usage, see http://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#specifying-more-general-memory-layouts

####################################################
# Single case (one neighborhood), single-threaded
####################################################

cdef int generic_fit_basic( int dimension, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D, double[::view.generic] fk, double[::1] xiManyD, double xi1D, double[::1] fi,
                                           double[::view.generic,::view.contiguous] sens, int do_sens, int order, long long knowns, int weighting_method, int debug ) nogil except -1

cdef int generic_fit_iterative( int dimension, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D, double[::view.generic] fk, double[::1] xiManyD, double xi1D, double[::1] fi,
                                               double[::view.generic,::view.contiguous] sens, int do_sens, int order, long long knowns, int weighting_method, int max_iter, int debug ) nogil except -1

####################################################
# Many cases, single-threaded
####################################################

# single-threaded
cdef int generic_fit_basic_many( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int debug ) nogil except -1

cdef int generic_fit_iterative_many( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                    double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                    double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                    double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                    int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter, int debug ) nogil except -1

####################################################
# Many cases, multithreaded
####################################################

cdef int generic_fit_basic_many_parallel( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                         double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                         double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                         double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                         int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int ntasks, int debug ) nogil except -1

cdef int generic_fit_iterative_many_parallel( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                             double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                             double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                             double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                             int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter, int ntasks, int debug ) nogil except -1

