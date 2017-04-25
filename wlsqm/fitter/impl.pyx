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

from __future__ import division, print_function, absolute_import

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, log10
from libc.math cimport fabs as c_abs

from numpy import asanyarray

cimport wlsqm.utils.lapackdrivers as drivers

cimport wlsqm.fitter.defs   as defs    # C constants
cimport wlsqm.fitter.infra  as infra   # centralized memory allocation infrastructure
cimport wlsqm.fitter.interp as interp  # interpolation of fitted model

####################################################
# Distance matrix (c) generation
####################################################

# (Here lowercase "c" is the name of the distance matrix; has no relation to the C programming language.)

# Adapter that takes in parameters for 1D, 2D and 3D cases, calls the appropriate routine and passes on the relevant parameters.
#
# This is used to work around the static typing in C to allow a single dimension-agnostic implementation for the high-level fitting routines.
#
cdef void make_c_nD( infra.Case* case, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D ) nogil:
    if case.dimension == 3:
        make_c_3D( case, xkManyD )
    elif case.dimension == 2:
        make_c_2D( case, xkManyD )
    else: # case.dimension == 1:
        make_c_1D( case, xk1D )


# Fill the "c" array, describing inter-point distances as needed in model fitting - 3D case
#
# case     : in, case metadata and pointers to allocated memory
#     order    : in, the order of the polynomial to be fitted
#     no       : in, the number of DOFs in the unreduced system (corresponding to dimension=3 and order)
#     nk       : in, number of neighbors (data points used in fit)
#     c        : out, size (nk,no) array (C-contiguous!) of inter-point distances and their various powers as needed in the fitting
#     w        : out, size (nk,) array of fitting weights for each point xk, based on a function of their squared distances from xi
#     weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
#                        different weightings are good for different use cases of WLSQM.
#     xi       : in, size (3,); the point xi itself
#
# xk       : in, size (nk,3); neighbor points; x[k,:] is the kth point
#
cdef void make_c_3D( infra.Case* case, double[::view.generic,::view.contiguous] xk ) nogil:

    # no-op in guest mode
    if not case.geometry_owned:
        return

    cdef double* c = case.c
    cdef double* w = case.w
    cdef int order = case.order
    cdef int weighting_method = case.weighting_method
    cdef int no    = case.no  # required size
    cdef int nk    = case.nk  # number of neighbors (data points used in fit) (usually the same as xk.shape[0])
    cdef double xi = case.xi
    cdef double yi = case.yi
    cdef double zi = case.zi

    cdef double d2, max_d2=0.  # euclidean distance squared

    cdef double dx,  dy,  dz   # x, y, z distances from the point xi
    cdef double dx2, dy2, dz2  # **2
    cdef double dx3, dy3, dz3  # **3

    # loop counters
    cdef int k       # neighbor points

    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    # generate the c^(j)_k array (see the documentation)
    #
    if order == 4:  # this is probably already pushing it (at double precision); may work for large enough region
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx  = xk[k,0] - xi  # signed x distance
            dy  = xk[k,1] - yi  # signed y distance
            dz  = xk[k,2] - zi  # signed z distance
            dx2 = dx*dx
            dy2 = dy*dy
            dz2 = dz*dz
            dx3 = dx2*dx
            dy3 = dy2*dy
            dz3 = dz2*dz

            d2 = dx2 + dy2 + dz2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2  # store the squared distances for now, we'll convert them to weights in a second pass (once we have  max_d2  which we need anyway)

            # Distance coefficients (these multiply the derivatives in the Taylor series)
            #
            c[ k*no + defs.i3_F_c    ] = 1.

            c[ k*no + defs.i3_X_c    ] = dx
            c[ k*no + defs.i3_Y_c    ] = dy
            c[ k*no + defs.i3_Z_c    ] = dz

            c[ k*no + defs.i3_X2_c   ] = 0.5 * dx2
            c[ k*no + defs.i3_XY_c   ] = dx*dy
            c[ k*no + defs.i3_Y2_c   ] = 0.5 * dy2
            c[ k*no + defs.i3_YZ_c   ] = dy*dz
            c[ k*no + defs.i3_Z2_c   ] = 0.5 * dz2
            c[ k*no + defs.i3_XZ_c   ] = dx*dz

            c[ k*no + defs.i3_X3_c   ] = onesixth * dx3
            c[ k*no + defs.i3_X2Y_c  ] = 0.5 * dx2*dy
            c[ k*no + defs.i3_XY2_c  ] = 0.5 * dx*dy2
            c[ k*no + defs.i3_Y3_c   ] = onesixth * dy3
            c[ k*no + defs.i3_Y2Z_c  ] = 0.5 * dy2*dz
            c[ k*no + defs.i3_YZ2_c  ] = 0.5 * dy*dz2
            c[ k*no + defs.i3_Z3_c   ] = onesixth * dz3
            c[ k*no + defs.i3_XZ2_c  ] = 0.5 * dx*dz2
            c[ k*no + defs.i3_X2Z_c  ] = 0.5 * dx2*dz
            c[ k*no + defs.i3_XYZ_c  ] = dx*dy*dz

            c[ k*no + defs.i3_X4_c   ] = one24th  * dx2*dx2
            c[ k*no + defs.i3_X3Y_c  ] = onesixth * dx3*dy
            c[ k*no + defs.i3_X2Y2_c ] = 0.25     * dx2*dy2
            c[ k*no + defs.i3_XY3_c  ] = onesixth * dx*dy3
            c[ k*no + defs.i3_Y4_c   ] = one24th  * dy2*dy2
            c[ k*no + defs.i3_Y3Z_c  ] = onesixth * dy3*dz
            c[ k*no + defs.i3_Y2Z2_c ] = 0.25     * dy2*dz2
            c[ k*no + defs.i3_YZ3_c  ] = onesixth * dy*dz3
            c[ k*no + defs.i3_Z4_c   ] = one24th  * dz2*dz2
            c[ k*no + defs.i3_XZ3_c  ] = onesixth * dx*dz3
            c[ k*no + defs.i3_X2Z2_c ] = 0.25     * dx2*dz2
            c[ k*no + defs.i3_X3Z_c  ] = onesixth * dx3*dz
            c[ k*no + defs.i3_X2YZ_c ] = 0.5      * dx2*dy*dz
            c[ k*no + defs.i3_XY2Z_c ] = 0.5      * dx*dy2*dz
            c[ k*no + defs.i3_XYZ2_c ] = 0.5      * dx*dy*dz2

    elif order == 3:
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx  = xk[k,0] - xi  # signed x distance
            dy  = xk[k,1] - yi  # signed y distance
            dz  = xk[k,2] - zi  # signed z distance
            dx2 = dx*dx
            dy2 = dy*dy
            dz2 = dz*dz

            d2 = dx2 + dy2 + dz2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            # Distance coefficients (these multiply the derivatives in the Taylor series)
            #
            c[ k*no + defs.i3_F_c   ] = 1.

            c[ k*no + defs.i3_X_c   ] = dx
            c[ k*no + defs.i3_Y_c   ] = dy
            c[ k*no + defs.i3_Z_c   ] = dz

            c[ k*no + defs.i3_X2_c  ] = 0.5 * dx2
            c[ k*no + defs.i3_XY_c  ] = dx*dy
            c[ k*no + defs.i3_Y2_c  ] = 0.5 * dy2
            c[ k*no + defs.i3_YZ_c  ] = dy*dz
            c[ k*no + defs.i3_Z2_c  ] = 0.5 * dz2
            c[ k*no + defs.i3_XZ_c  ] = dx*dz

            c[ k*no + defs.i3_X3_c  ] = onesixth * dx2*dx
            c[ k*no + defs.i3_X2Y_c ] = 0.5 * dx2*dy
            c[ k*no + defs.i3_XY2_c ] = 0.5 * dx*dy2
            c[ k*no + defs.i3_Y3_c  ] = onesixth * dy*dy2
            c[ k*no + defs.i3_Y2Z_c ] = 0.5 * dy2*dz
            c[ k*no + defs.i3_YZ2_c ] = 0.5 * dy*dz2
            c[ k*no + defs.i3_Z3_c  ] = onesixth * dz*dz2
            c[ k*no + defs.i3_XZ2_c ] = 0.5 * dx*dz2
            c[ k*no + defs.i3_X2Z_c ] = 0.5 * dx2*dz
            c[ k*no + defs.i3_XYZ_c ] = dx*dy*dz

    elif order == 2:
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            # Distances from point i to the neighbor points
            #
            dx  = xk[k,0] - xi  # signed x distance
            dy  = xk[k,1] - yi  # signed y distance
            dz  = xk[k,2] - zi  # signed z distance
            dx2 = dx*dx
            dy2 = dy*dy
            dz2 = dz*dz

            d2 = dx2 + dy2 + dz2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i3_F_c  ] = 1.

            c[ k*no + defs.i3_X_c  ] = dx
            c[ k*no + defs.i3_Y_c  ] = dy
            c[ k*no + defs.i3_Z_c  ] = dz

            c[ k*no + defs.i3_X2_c ] = 0.5 * dx2
            c[ k*no + defs.i3_XY_c ] = dx*dy
            c[ k*no + defs.i3_Y2_c ] = 0.5 * dy2
            c[ k*no + defs.i3_YZ_c ] = dy*dz
            c[ k*no + defs.i3_Z2_c ] = 0.5 * dz2
            c[ k*no + defs.i3_XZ_c ] = dx*dz

    elif order == 1:
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx  = xk[k,0] - xi  # signed x distance
            dy  = xk[k,1] - yi  # signed y distance
            dz  = xk[k,2] - zi  # signed z distance

            d2 = dx*dx + dy*dy + dz*dz
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i3_F_c ] = 1.

            c[ k*no + defs.i3_X_c ] = dx
            c[ k*no + defs.i3_Y_c ] = dy
            c[ k*no + defs.i3_Z_c ] = dz

    else: # order == 0:
        # even in this case, we need the distances to compute the weights.
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx  = xk[k,0] - xi  # signed x distance
            dy  = xk[k,1] - yi  # signed y distance
            dz  = xk[k,2] - zi  # signed z distance

            d2 = dx*dx + dy*dy + dz*dz
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i3_F_c ] = 1.

    # Convert squared distances to weights.
    #
    infra.Case_make_weights( case, max_d2 )


# Fill the "c" array, describing inter-point distances as needed in model fitting - 2D case
#
# case     : in, case metadata and pointers to allocated memory
#     order    : in, the order of the polynomial to be fitted
#     no       : in, the number of DOFs in the unreduced system (corresponding to dimension=2 and order)
#     nk       : in, number of neighbors (data points used in fit)
#     c        : out, size (nk,no) array (C-contiguous!) of inter-point distances and their various powers as needed in the fitting
#     w        : out, size (nk,) array of fitting weights for each point xk, based on a function of their squared distances from xi
#     weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
#                        different weightings are good for different use cases of WLSQM.
#     xi       : in, size (2,); the point xi itself
#
# xk       : in, size (nk,2); neighbor points; x[k,:] is the kth point
#
cdef void make_c_2D( infra.Case* case, double[::view.generic,::view.contiguous] xk ) nogil:

    # no-op in guest mode
    if not case.geometry_owned:
        return

    cdef double* c = case.c
    cdef double* w = case.w
    cdef int order = case.order
    cdef int weighting_method = case.weighting_method
    cdef int no    = case.no  # required size
    cdef int nk    = case.nk  # number of neighbors (data points used in fit) (usually the same as xk.shape[0])
    cdef double xi = case.xi
    cdef double yi = case.yi

    cdef double d2, max_d2=0.  # euclidean distance squared

    cdef double dx,  dy   # x, y distances from the point xi
    cdef double dx2, dy2  # **2
    cdef double dx3, dy3  # **3

    # loop counters
    cdef int k       # neighbor points

    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    # generate the c^(j)_k array (see the documentation)
    #
    if order == 4:  # this is probably already pushing it (at double precision); may work for large enough region
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx = xk[k,0] - xi  # signed x distance
            dy = xk[k,1] - yi  # signed y distance
            dx2 = dx*dx
            dy2 = dy*dy
            dx3 = dx2*dx
            dy3 = dy2*dy

            d2 = dx2 + dy2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2  # store the squared distances for now, we'll convert them to weights in a second pass (once we have  max_d2  which we need anyway)

            # Distance coefficients (these multiply the derivatives in the Taylor series)
            #
            c[ k*no + defs.i2_F_c    ] = 1.         # c^(0)_k  in the documentation

            c[ k*no + defs.i2_X_c    ] = dx         # c^(1)_k
            c[ k*no + defs.i2_Y_c    ] = dy         # c^(2)_k

            c[ k*no + defs.i2_X2_c   ] = 0.5 * dx2  # c^(3)_k
            c[ k*no + defs.i2_XY_c   ] = dx*dy      # c^(4)_k
            c[ k*no + defs.i2_Y2_c   ] = 0.5 * dy2  # c^(5)_k

            c[ k*no + defs.i2_X3_c   ] = onesixth * dx3
            c[ k*no + defs.i2_X2Y_c  ] = 0.5 * dx2*dy
            c[ k*no + defs.i2_XY2_c  ] = 0.5 * dx*dy2
            c[ k*no + defs.i2_Y3_c   ] = onesixth * dy3

            c[ k*no + defs.i2_X4_c   ] = one24th  * dx2*dx2
            c[ k*no + defs.i2_X3Y_c  ] = onesixth * dx3*dy
            c[ k*no + defs.i2_X2Y2_c ] = 0.25     * dx2*dy2
            c[ k*no + defs.i2_XY3_c  ] = onesixth * dx*dy3
            c[ k*no + defs.i2_Y4_c   ] = one24th  * dy2*dy2

    elif order == 3:
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx = xk[k,0] - xi  # signed x distance
            dy = xk[k,1] - yi  # signed y distance
            dx2 = dx*dx
            dy2 = dy*dy

            d2 = dx2 + dy2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            # Distance coefficients (these multiply the derivatives in the Taylor series)
            #
            c[ k*no + defs.i2_F_c   ] = 1.         # c^(0)_k  in the documentation

            c[ k*no + defs.i2_X_c   ] = dx         # c^(1)_k
            c[ k*no + defs.i2_Y_c   ] = dy         # c^(2)_k

            c[ k*no + defs.i2_X2_c  ] = 0.5 * dx2  # c^(3)_k
            c[ k*no + defs.i2_XY_c  ] = dx*dy      # c^(4)_k
            c[ k*no + defs.i2_Y2_c  ] = 0.5 * dy2  # c^(5)_k

            c[ k*no + defs.i2_X3_c  ] = onesixth * dx2*dx
            c[ k*no + defs.i2_X2Y_c ] = 0.5 * dx2*dy
            c[ k*no + defs.i2_XY2_c ] = 0.5 * dx*dy2
            c[ k*no + defs.i2_Y3_c  ] = onesixth * dy*dy2

    elif order == 2:
        for k in range(nk):
            dx = xk[k,0] - xi  # signed x distance
            dy = xk[k,1] - yi  # signed y distance
            dx2 = dx*dx
            dy2 = dy*dy

            d2 = dx2 + dy2
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i2_F_c  ] = 1.         # c^(0)_k  in the documentation

            c[ k*no + defs.i2_X_c  ] = dx         # c^(1)_k
            c[ k*no + defs.i2_Y_c  ] = dy         # c^(2)_k

            c[ k*no + defs.i2_X2_c ] = 0.5 * dx2  # c^(3)_k
            c[ k*no + defs.i2_XY_c ] = dx*dy      # c^(4)_k
            c[ k*no + defs.i2_Y2_c ] = 0.5 * dy2  # c^(5)_k

    elif order == 1:
        for k in range(nk):
            dx = xk[k,0] - xi  # signed x distance
            dy = xk[k,1] - yi  # signed y distance

            d2 = dx*dx + dy*dy
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i2_F_c ] = 1.         # c^(0)_k  in the documentation

            c[ k*no + defs.i2_X_c ] = dx         # c^(1)_k
            c[ k*no + defs.i2_Y_c ] = dy         # c^(2)_k

    else: # order == 0:
        # even in this case, we need the distances to compute the weights.
        for k in range(nk):
            dx = xk[k,0] - xi  # signed x distance
            dy = xk[k,1] - yi  # signed y distance

            d2 = dx*dx + dy*dy
            if d2 > max_d2:
                max_d2 = d2
            w[k] = d2

            c[ k*no + defs.i2_F_c ] = 1.         # c^(0)_k  in the documentation

    # Convert squared distances to weights.
    #
    infra.Case_make_weights( case, max_d2 )


# Fill the "c" array, describing inter-point distances as needed in model fitting - 1D case
#
# case     : in, case metadata and pointers to allocated memory
#     order    : in, the order of the polynomial to be fitted
#     no       : in, the number of DOFs in the unreduced system (corresponding to dimension=1 and order)
#     nk       : in, number of neighbors (data points used in fit)
#     c        : out, size (nk,no) array (C-contiguous!) of inter-point distances and their various powers as needed in the fitting
#     w        : out, size (nk,) array of fitting weights for each point xk, based on a function of their squared distances from xi
#     weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
#                        different weightings are good for different use cases of WLSQM.
#     xi       : in, the point xi itself
#
# xk       : in, size (nk,); neighbor points; x[k] is the kth point
#
cdef void make_c_1D( infra.Case* case, double[::view.generic] xk ) nogil:

    # no-op in guest mode
    if not case.geometry_owned:
        return

    cdef double* c = case.c
    cdef double* w = case.w
    cdef int order = case.order
    cdef int weighting_method = case.weighting_method
    cdef int no    = case.no  # required size
    cdef int nk    = case.nk  # number of neighbors (data points used in fit) (usually the same as xk.shape[0])
    cdef double xi = case.xi

    cdef double max_d2=0.  # euclidean distance squared (largest seen)
    cdef double dx, dx2

    # loop counters
    cdef int k       # neighbor points

    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    # generate the c^(j)_k array (see the documentation)
    #
    if order == 4:  # this is probably already pushing it (at double precision); may work for large enough region
        for k in range(nk):
            # Distances from point i to the neighbor points
            #
            dx  = xk[k] - xi  # signed x distance
            dx2 = dx*dx

            if dx2 > max_d2:
                max_d2 = dx2
            w[k] = dx2  # store the squared distances for now, we'll convert them to weights in a second pass (once we have  max_d2  which we need anyway)

            # Distance coefficients (these multiply the derivatives in the Taylor series)
            #
            c[ k*no + defs.i1_F_c  ] = 1.
            c[ k*no + defs.i1_X_c  ] = dx
            c[ k*no + defs.i1_X2_c ] = 0.5 * dx2
            c[ k*no + defs.i1_X3_c ] = onesixth * dx*dx2
            c[ k*no + defs.i1_X4_c ] = one24th  * dx2*dx2

    elif order == 3:
        for k in range(nk):
            dx  = xk[k] - xi
            dx2 = dx*dx

            if dx2 > max_d2:
                max_d2 = dx2
            w[k] = dx2

            c[ k*no + defs.i1_F_c  ] = 1.
            c[ k*no + defs.i1_X_c  ] = dx
            c[ k*no + defs.i1_X2_c ] = 0.5 * dx2
            c[ k*no + defs.i1_X3_c ] = onesixth * dx*dx2

    elif order == 2:
        for k in range(nk):
            dx  = xk[k] - xi
            dx2 = dx*dx

            if dx2 > max_d2:
                max_d2 = dx2
            w[k] = dx2

            c[ k*no + defs.i1_F_c  ] = 1.
            c[ k*no + defs.i1_X_c  ] = dx
            c[ k*no + defs.i1_X2_c ] = 0.5 * dx2

    elif order == 1:
        for k in range(nk):
            dx  = xk[k] - xi
            dx2 = dx*dx

            if dx2 > max_d2:
                max_d2 = dx2
            w[k] = dx2

            c[ k*no + defs.i1_F_c ] = 1.
            c[ k*no + defs.i1_X_c ] = dx

    else: # order == 0:
        # even in this case, we need the distances to compute the weights.
        for k in range(nk):
            dx  = xk[k] - xi
            dx2 = dx*dx

            if dx2 > max_d2:
                max_d2 = dx2
            w[k] = dx2

            c[ k*no + defs.i1_F_c ] = 1.

    # Convert squared distances to weights.
    #
    infra.Case_make_weights( case, max_d2 )


####################################################
# Problem matrix (A) generation
####################################################

# These routines generate the problem matrix A, based on the distance matrix C.

# Create the problem matrix A (naively, without scaling - see also preprocess_A()).
#
# case   : in, case metadata and pointers to allocated memory
#     A      : out, (nr, nr), Fortran-contiguous, the matrix A
#     c      : in, (nk,no) coefficient array, as output from make_c_?D()
#     w      : in, (nk,) array of weights, as output from make_c_?D()
#     nk     : in, number of neighbors (data points used in fit)
#     no     : in, number of DOFs in the original unreduced system, as output from make_c_?D()
#     nr     : in, number of DOFs in the reduced system, as output from remap()
#     knowns : in, knowns bitmask
#     o2r    : out, array of size (no,), will contain DOF mapping original --> reduced   (must be allocated by caller!)
#     r2o    : out, array of size (no,), will contain DOF mapping reduced  --> original  (must be allocated by caller!)
#
cdef void make_A( infra.Case* case ) nogil:

    # no-op in guest mode
    if not case.geometry_owned:
        return

    # all tagged as knowns --> nothing to solve
    #
    if case.nr < 1:
        return  # can't use GIL and raise exception, since we'll be called in the OpenMP threads

    cdef double* A        = case.A  # (nr, nr), Fortran-contiguous
    cdef double* c        = case.c
    cdef double* w        = case.w
    cdef int nk           = case.nk
    cdef int nr           = case.nr
    cdef int no           = case.no
    cdef long long knowns = case.knowns
    cdef int* o2r         = case.o2r
    cdef int* r2o         = case.r2o

    # construct the reduced system (naively, without scaling; preconditioning is applied later in preprocess_A())
    #
    cdef double acc           # accumulator for summation
    cdef int j, oj, m, om, k  # loop counters
    for j in range(nr):      # row in reduced system
        oj = r2o[j]  # index of reduced DOF j in original system

        # fill reduced matrix A with the coefficients of the reduced part
        #
        for m in range(nr):  # col in reduced system
            om = r2o[m]  # index of reduced DOF m in original system
            acc = 0.
            for k in range(nk):
                # c is indexed using indices of the original system (and is C-contiguous)
                acc += w[k] * c[ k*no + om ] * c[ k*no + oj ]  # apply w[k], the fitting weight of the error component k
            A[j + nr*m]     = acc  # A is indexed using indices of the reduced system (and is Fortran-contiguous)

# Precondition (scale) and LU factorize A.
#
# case      : in, case metadata and pointers to allocated memory
#     A         : in/out. Upon enter, the original matrix A as output by make_A().
#                         Upon exit,  the (packed) LU factorization of the preconditioned A.
#     nr        : in, number of DOFs in the reduced system. ("A" is of size (nr,nr))
#     row_scale : out, (nr,) array of row scaling factors  (must be allocated by caller!)
#                      When solving, this is needed to scale the RHS accordingly.
#     col_scale : out, (nr,) array of column scaling factors  (must be allocated by caller!)
#                      When solving, this is needed to scale the solution.
#     o2r       : in, array of size (no,), DOF mapping original --> reduced, as output by make_A()   (only used if debug=True)
#     r2o       : in, array of size (no,), DOF mapping reduced  --> original, as output by make_A()  (only used if debug=True)
#
# debug     : in, boolean. If debug is True, compute the condition number (2-norm directly via SVD) of the original and scaled A.
#             The condition numbers are written in case.cond_orig and case.cond_scaled.
#
cdef void preprocess_A( infra.Case* case, int debug ) nogil:

    # no-op in guest mode
    if not case.geometry_owned:
        return

    cdef double* A         = case.A
    cdef int* ipiv         = case.ipiv
    cdef double* row_scale = case.row_scale
    cdef double* col_scale = case.col_scale
    cdef int nr            = case.nr
    cdef int* o2r          = case.o2r
    cdef int* r2o          = case.r2o

    # all tagged as knowns --> nothing to solve
    #
    if nr < 1:
        return  # can't use GIL and raise exception, since we'll be called in the OpenMP threads

    # We may need a copy of A for computing condition numbers.
    #
    cdef double* Acopy = <double*>0
    if debug:
        Acopy = <double*>malloc( nr*nr*sizeof(double) )  # bypass the custom allocator; debug mode is not used in production code where memory fragmentation may matter
        drivers.copygeneral_c( Acopy, A, nr, nr )

    # Use scaling to improve the condition number of A, resulting in more correct digits in the solution.
    # The effects are drastic especially in high-order fitting.
    #
    # Specifically, we use DGESV, which works by LU factorization with pivoting. For this algorithm the
    # relative accuracy is O(kappa(A) * machine_epsilon), where at double precision, machine_epsilon ~ 1e-16.
    # ( http://scicomp.stackexchange.com/questions/19289/are-direct-solvers-affect-by-the-condition-number-of-a-matrix )
    #
    cdef int iterations_taken = drivers.rescale_ruiz2001_c( A, nr, nr, row_scale, col_scale )  # compute row and column scaling factors
    drivers.apply_scaling_c( A, nr, nr, row_scale, col_scale )     # using the computed factors, scale A in-place

    # Compute the condition numbers before factorizing A.
    #
    # We can't use dgecon() (which would use the factorized A), because we want the 2-norm condition number and dgecon() only supports 1-norm and infinity-norm.
    #
    cdef double* S = <double*>0
    cdef double cond_orig, cond_scaled
    cdef int j
    if debug:
        # Estimate the condition number (before and after scaling) and save it to the Case instance.
        #
        # According to https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.cond.html
        # NumPy's cond() computes the 2-norm condition number (default) directly from the SVD.
        # http://mathworld.wolfram.com/ConditionNumber.html says this is max(S)/min(S).
        #
        # We can use LAPACK's dgesvd() to do this at the C level.
        # The output is sorted so that S[i] >= S[i+1].

        # original matrix
        S = <double*>malloc( nr*sizeof(double) )  # bypass the custom allocator; debug mode is not used in production code where memory fragmentation may matter
        drivers.svd_c( Acopy, nr, nr, S )  # result goes into S; Acopy is destroyed (overwritten)
        case.cond_orig   = S[0] / S[nr-1]

        # scaled matrix
        drivers.copygeneral_c( Acopy, A, nr, nr )
        drivers.svd_c( Acopy, nr, nr, S )
        case.cond_scaled = S[0] / S[nr-1]

        free( <void*>S )

    # Compute LU factorization of scaled A.
    #
    drivers.generalfactor_c( A, ipiv, nr )  # TODO: error handling (can't use GIL and catch exception, since we'll be called in the OpenMP threads)

    if debug:
        free( <void*>Acopy )  # bypassing the custom allocator, stuff for debug mode only


####################################################
# RHS handling and solving
####################################################

# Perform model fitting for given data.
#
# case      : in, case metadata and pointers to allocated memory
#     no        : in, number of DOFs in the original (unreduced) system, as output by make_c_?D()
#     c         : coefficient array of size (nk,no), as output by make_c_?D()
#     w         : in, (nk,) array of weights, as output from make_c_?D()
#
#     nr        : in, number of DOFs in the reduced system. ("A" is of size (nr,nr)), as output by make_A()
#     o2r       : in, array of size (no,), DOF mapping original --> reduced, as output by make_A()
#     r2o       : in, array of size (no,), DOF mapping reduced  --> original, as output by make_A()
#
#     A         : in, the (packed) LU factorization of the preconditioned A, as output by preprocess_A().
#     ipiv      : in, pivot information for solver, as output by preprocess_A().
#     row_scale : in, row scaling factors for scaling RHS, as output by preprocess_A().
#     col_scale : in, column scaling factors for scaling solution, as output by preprocess_A().
#
#     fi        : in/out, passed through from fit_*(), function data at xi; knowns in, unknowns out
#                 The number of elements in fi is taken from case.no.
#
#     knowns    : in, passed through from fit_*()
#
#     wrk       : work space, must be allocated by caller. (done by Case_new())
#                 If do_sens=False, must have space for (nr,) doubles.
#                 If do_sens=True,  must have space for (nr,) + (nr,nk) = (nk+1)*nr doubles.
#
# fk        : in, passed through from fit_*(), function value data at the neighbor points
#
# sens      : out, passed through from fit_*()
# do_sens   : in, boolean: whether to perform sensitivity analysis. If False, "sens" can be none.
#             (if the Case object has its do_sens flag set, it *supports* sensitivity analysis;
#              this flag determines whether we actually *do* it.)
#
# taskid    : in, number of parallel processing task (0, 1, ..., ntasks-1)
#             Used to get access to the RHS work space.
#
cdef void solve( infra.Case* case, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid ) nogil:

    cdef double nan = 0./0.  # FIXME: a better way to get NaN than to abuse IEEE-754 specification?

    cdef int no            = case.no
    cdef int nr            = case.nr
    cdef int nk            = case.nk  # usually the same as fk.shape[0]  # number of neighbors (data points used in fit)

    # all tagged as knowns --> nothing to solve (important; LAPACK doesn't like N = 0)
    #
    if nr < 1:
        return

    cdef long long knowns  = case.knowns

    cdef double* c         = case.c
    cdef double* w         = case.w
    cdef int* o2r          = case.o2r
    cdef int* r2o          = case.r2o

    cdef double* LU        = case.A  # case.A has already been factored when solve() is called
    cdef int* ipiv         = case.ipiv
    cdef double* row_scale = case.row_scale
    cdef double* col_scale = case.col_scale

    cdef double* fi        = case.fi

    cdef double* wrk       = infra.Case_get_wrk( case, taskid )

    cdef double* b = &wrk[0]   # (nr,)
    cdef double* s = &wrk[nr]  # the rest of the work space; (nr, nk), Fortran-contiguous

    # construct the RHS of the reduced system, applying also row scaling
    #
    cdef double acc    # accumulator for summation
    cdef int j, oj, k  # loop counters
    if do_sens:
        for j in range(nr):      # row in reduced system
            oj = r2o[j]  # index of reduced DOF j in original system

            acc = 0.
            for k in range(nk):
                acc += w[k] * fk[k] * c[ k*no + oj ]  # apply also w[k], the fitting weight of error component k

                # sensitivity of the solution w.r.t. fk[k]
                # (only one term in each row; this is d(b) / d( fk[k] ), leaving only c^(j)_k)
                s[j + nr*k] = row_scale[j] * w[k] * c[ k*no + oj ]
            b[j] = row_scale[j] * acc
    else:
        for j in range(nr):      # row in reduced system
            oj = r2o[j]  # index of reduced DOF j in original system

            acc = 0.
            for k in range(nk):
                acc += w[k] * fk[k] * c[ k*no + oj ]   # apply also w[k], the fitting weight of error component k
            b[j] = row_scale[j] * acc

    # apply algebraic elimination to the knowns
    #
    cdef int om
    for om in range(no):  # DOF in original system
        # handle only the eliminated DOFs
        if knowns & (1LL << om):
            # om now represents an eliminated column of the original system

            # update all rows of b in the reduced system.
            #
            # We must account for row scaling, because the whole row is scaled, and hence b, too.
            #
            # But column scaling cancels out for eliminated DOFs:
            #
            #   fi[om] * c[ k*SIZE + om ] = (fi[om] * col_scale) * (c[ k*SIZE + om ] / col_scale)
            #   ======   ================   --------------------   ------------------------------
            #   "small"    original coeff   "big" (scaled) units   scaled (big) coeff in A
            #   (orig.)
            #    units
            #
            # or intuitively:
            #
            #   many small units * small coeff = few big units * scaled (big) coeff
            #
            # This is fortunate, because the column scaling factor has not been computed for the eliminated columns!
            #
            for j in range(nr):  # row in reduced system
                oj = r2o[j]  # index of reduced DOF j in original system
                for k in range(nk):
                    b[j] -= fi[om] * w[k] * c[ k*no + om ] * c[ k*no + oj ] * row_scale[j]   # apply also w[k], the fitting weight of error component k

            # also, we may as well do this here:
            if do_sens:  # no is rather small, not many extra instructions
                for k in range(nk):
                    sens[k, om] = nan  # no sensitivity data for eliminated DOFs

    # solve (overwriting b with solution)
    #
    drivers.generalfactored_c( LU, ipiv, b, nr )

    # perform sensitivity analysis (overwriting s with the sensitivity data)
    #
    if do_sens:  # if no sensitivity analysis, major runtime savings occur here
        # we must loop as there is no "generalfactoreds_c()".
        for k in range(nk):
            drivers.generalfactored_c( LU, ipiv, &s[k*nr], nr )

    # extract solution (filling in only the unknowns in fi[])
    #
    for j in range(nr):  # row in reduced system
        oj = r2o[j]  # index of reduced DOF j in original system

        fi[oj] = b[j] * col_scale[j]  # compute original x from the column-scaled x

        # fill sensitivity data w.r.t. each unknown j and each fk[k]
        if do_sens:  # again, small nr, not many extra instructions
            for k in range(nk):
                sens[k, oj] = s[nr*k + j] * col_scale[j]


# TODO/FIXME: This function is identical to solve() except the signature. We do this senseless code duplication because solve_iterative() needs a version with raw pointers to contiguous C-level buffers to avoid GIL.
#
# TODO/FIXME: Memoryviews can be *accessed* without the GIL; the problem is that *creating* memoryview objects requires the GIL. The function solve_iterative() can't afford to do that,
# TODO/FIXME: because it is a low-level function that may be (and actually is; see the .pyx source for wlsqm.fitter.expert) called from an OpenMP parallel loop with the GIL released.
#
# Note that this version allows one to use a separate array fi, which is not necessarily the one stored in the Case instance. solve_iterative() uses this feature to compute the iterative updates.
#
# This function is not exported.
#
# - the number of elements in fk is taken from case.nk
# - the number of elements in fi is taken from case.no
#
cdef void solve_contig( infra.Case* case, double* fk, double* fi, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid ) nogil:

    cdef double nan = 0./0.  # FIXME: a better way to get NaN than to abuse IEEE-754 specification?

    cdef int no            = case.no
    cdef int nr            = case.nr
    cdef int nk            = case.nk  # usually the same as fk.shape[0]  # number of neighbors (data points used in fit)

    # all tagged as knowns --> nothing to solve (important; LAPACK doesn't like N = 0)
    #
    if nr < 1:
        return

    cdef long long knowns  = case.knowns

    cdef double* c         = case.c
    cdef double* w         = case.w
    cdef int* o2r          = case.o2r
    cdef int* r2o          = case.r2o

    cdef double* LU        = case.A  # case.A has already been factored when solve() is called
    cdef int* ipiv         = case.ipiv
    cdef double* row_scale = case.row_scale
    cdef double* col_scale = case.col_scale

    cdef double* wrk       = infra.Case_get_wrk( case, taskid )

    cdef double* b = &wrk[0]   # (nr,)
    cdef double* s = &wrk[nr]  # the rest of the work space; (nr, nk), Fortran-contiguous

    # construct the RHS of the reduced system, applying also row scaling
    #
    cdef double acc    # accumulator for summation
    cdef int j, oj, k  # loop counters
    if do_sens:
        for j in range(nr):      # row in reduced system
            oj = r2o[j]  # index of reduced DOF j in original system

            acc = 0.
            for k in range(nk):
                acc += w[k] * fk[k] * c[ k*no + oj ]  # apply also w[k], the fitting weight of error component k

                # sensitivity of the solution w.r.t. fk[k]
                # (only one term in each row; this is d(b) / d( fk[k] ), leaving only c^(j)_k)
                s[j + nr*k] = row_scale[j] * w[k] * c[ k*no + oj ]
            b[j] = row_scale[j] * acc
    else:
        for j in range(nr):      # row in reduced system
            oj = r2o[j]  # index of reduced DOF j in original system

            acc = 0.
            for k in range(nk):
                acc += w[k] * fk[k] * c[ k*no + oj ]   # apply also w[k], the fitting weight of error component k
            b[j] = row_scale[j] * acc

    # apply algebraic elimination to the knowns
    #
    cdef int om
    for om in range(no):  # DOF in original system
        # handle only the eliminated DOFs
        if knowns & (1LL << om):
            # om now represents an eliminated column of the original system

            # update all rows of b in the reduced system.
            #
            # We must account for row scaling, because the whole row is scaled, and hence b, too.
            #
            # But column scaling cancels out for eliminated DOFs:
            #
            #   fi[om] * c[ k*SIZE + om ] = (fi[om] * col_scale) * (c[ k*SIZE + om ] / col_scale)
            #   ======   ================   --------------------   ------------------------------
            #   "small"    original coeff   "big" (scaled) units   scaled (big) coeff in A
            #   (orig.)
            #    units
            #
            # or intuitively:
            #
            #   many small units * small coeff = few big units * scaled (big) coeff
            #
            # This is fortunate, because the column scaling factor has not been computed for the eliminated columns!
            #
            for j in range(nr):  # row in reduced system
                oj = r2o[j]  # index of reduced DOF j in original system
                for k in range(nk):
                    b[j] -= fi[om] * w[k] * c[ k*no + om ] * c[ k*no + oj ] * row_scale[j]   # apply also w[k], the fitting weight of error component k

            # also, we may as well do this here:
            if do_sens:  # no is rather small, not many extra instructions
                for k in range(nk):
                    sens[k, om] = nan  # no sensitivity data for eliminated DOFs

    # solve (overwriting b with solution)
    #
    drivers.generalfactored_c( LU, ipiv, b, nr )

    # perform sensitivity analysis (overwriting s with the sensitivity data)
    #
    if do_sens:  # if no sensitivity analysis, major runtime savings occur here
        # we must loop as there is no "generalfactoreds_c()".
        for k in range(nk):
            drivers.generalfactored_c( LU, ipiv, &s[k*nr], nr )

    # extract solution (filling in only the unknowns in fi[])
    #
    for j in range(nr):  # row in reduced system
        oj = r2o[j]  # index of reduced DOF j in original system

        fi[oj] = b[j] * col_scale[j]  # compute original x from the column-scaled x

        # fill sensitivity data w.r.t. each unknown j and each fk[k]
        if do_sens:  # again, small nr, not many extra instructions
            for k in range(nk):
                sens[k, oj] = s[nr*k + j] * col_scale[j]


# Algorithm with iterative refinement of the fit to mitigate roundoff.
#
# Uses solve() to do the actual solving.
#
# Note that the xk data (the same neighbor points as used in the fitting) must be passed in here, so that the fitted model can be interpolated to the correct points when computing the error.
# The model origin xi is stored in the Case object, but it would make no sense to store the xk data in the Case, because this data is in practice likely shared between problem instances.
#
# In any case, the data stored in the Case object is dependent on xk and xi not changing on the fly. If xi or xk suddenly change, the c and A matrices must be re-generated.
#
cdef int solve_iterative( infra.Case* case, double[::view.generic] fk, double[::view.generic,::view.contiguous] sens, int do_sens, int taskid, int max_iter,
                          double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D ) nogil:

    DEF DONT_DO_SENS = 0  # value of do_sens when we don't want to do sensitivity analysis (used in the iterative refinement step)
    DEF DIFF = 0  # interpolate function value (not a derivative)

    # First, as usual, fit the model against the original user-given data fk[], automatically updating case.fi:
    #
    solve( case, fk, sens, do_sens, taskid )

    # Then refine iteratively to reduce numerical error.

    # Get temporary arrays for work space, to be used for fitting the remaining error.
    cdef double* wrk_fk = infra.Case_get_fk_tmp( case, taskid )
    cdef double* wrk_fi = infra.Case_get_fi_tmp( case, taskid )
    cdef int nk = case.nk, no = case.no

    # Zero out those coefficients of the "error reduction" fit for which the solution is known exactly,
    # since we cannot change those to reduce the error. (They will be tagged as knowns for the
    # error reduction problem, too.)
    #
    cdef long long knowns = case.knowns
    cdef int om
    for om in range(no):
        if knowns & (1LL << om):
            wrk_fi[om] = 0.  # this component of the solution has no error to reduce

    cdef int i=0, k  # the user may have accidentally given max_iter=0 so we must initialize i here
    cdef double tmp, norm, prev_norm=-1.
    cdef int dimension = case.dimension
    cdef int order     = case.order
    cdef double* fi    = case.fi  # the array has case.no elements
#    DEF epsilon = 1e-15
    for i in range(max_iter):
        # Using the latest fi[], interpolate the fitted model to the points xk, overwriting wrk_fk.
        #
        # This sets up the new target for fitting (i.e. remaining error at the points xk).
        #
        interp.interpolate_nD( case, xkManyD, xk1D, wrk_fk, DIFF )  # only either *ManyD or *1D are used, depending on case.dimension
        for k in range(nk):
            wrk_fk[k] = fk[k] - wrk_fk[k]  # overwrite wrk_fk a second time, now with the remaining error


        # Convergence check: monitor the l-infinity norm (max abs) of remaining error
        #
        # Note that if we get very lucky (and the data we are fitting against allows it),
        # at i=0, the error may already be below epsilon.
        #
        # In such a case we break immediately before performing any error reduction fit;
        # thus the optimal ordering is to check convergence *before* actually fitting.
        #
        # TODO: What is a good error measure? |f| may be very large, or it may be near (or at!) zero...
        # TODO: Or is it better to always run, say, 3 iterations and be done with it? (would simplify the code)
        #
        norm = c_abs(wrk_fk[0])
        for k in range(1,nk):
            tmp = c_abs(wrk_fk[k])
            if tmp > norm:
                norm = tmp

#        # This is useless as it depends on the scaling of the problem (size of function values fk).
#        #
#        # Checking that the l-infinity norm of the error no longer changes should be good enough also in cases where the iteration converges to machine accuracy.
#        #
#        # If we really want to do this, we should probably normalize by the l-infinity norm of fk. But that runs into trouble if all fk are very small.
#        if norm <= epsilon:  # converged, error successfully eliminated
#            break

        # If the norm is the same as in the previous iteration (down to floating point equality!),
        # we have probably stagnated (i.e. the remaining error is no longer changing).
        #
        # In this case iterative refinement does not reduce the error any further (the remaining error is likely to be modeling error, not roundoff!),
        # and we can consider the fitting complete.
        #
        if norm == prev_norm:  # converged, further corrections do nothing
                               # (when we get here, we have done the error reduction fit at least once and the norm didn't change)
                               # (the initial prev_norm is an invalid value so this cannot trigger at the first iteration)
            break
        prev_norm = norm


        # Fit the coefficients wrk_fi to model the remaining error wrk_fk,
        # using the same type of model as used for the original fit.
        #
        # To avoid the need for GIL, use the solver version that accepts raw pointers for fk and fi.
        #
        # Note that for the coefficient array, we use wrk_fi, not case.fi! This is important to keep the original solution intact.
        #
        solve_contig( case, wrk_fk, wrk_fi, sens, DONT_DO_SENS, taskid )

        # Update the unknown components of the solution fi
        # to make the model describe  original data + error .
        #
        for om in range(no):
            if not (knowns & (1LL << om)):
                fi[om] += wrk_fi[om]

    else:  # if the loop didn't break, the last iteration of the loop contributed one more refinement iteration
        i += 1

    return i  # number of refinement iterations taken (i, not i+1, because the fitting code comes *after* the convergence check)

