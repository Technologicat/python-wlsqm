# -*- coding: utf-8 -*-
#
# Set Cython compiler directives. This section must appear before any code!
#
# For available directives, see:
#
# http://docs.cython.org/en/latest/src/reference/compilation.html
#
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

This module contains the simple API. For advanced features, see the module wlsqm.fitter.expert.

JJ 2016-11-07
"""

# ------------------------------- notes -------------------------------
#
# - each local problem may have different nk (number of neighbor points i.e. points, in the point cloud, other than the reference point xi)
#   - we must support this, since in practice there is no guarantee about the number of neighbors at distance <= r (considering irregular clouds of points).
#   - at the Python end, we can allocate fk using the maximum size, and then supply a separate array nk[] of size (nprob,) that gives the actual used size for each problem instance.
#   - this allows us to allocate the correct size for each problem (from the buffer), not wasting any memory at the C end.
#
# - no contiguous buffer in the prep stage to allow generating the input by slicing an array (very useful); but the generated matrices will be contiguous!
#   --> need to read only the function value data (fk[]) non-contiguously in the solve stage

from __future__ import division
from __future__ import absolute_import

from libc.stdlib cimport malloc, free

cimport cython.parallel
cimport openmp

from libc.math cimport fabs as c_abs

cimport wlsqm.fitter.defs  as defs   # C constants
cimport wlsqm.fitter.infra as infra  # centralized memory allocation infrastructure
cimport wlsqm.fitter.impl  as impl   # low-level routines (implementation)

####################################################
# Python API
####################################################

##########
# 3D case
##########

def fit_3D( double[::view.generic,::view.contiguous] xk, double[::view.generic] fk, double[::1] xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b3_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int debug=0 ):
    """Fit one local model to 3D scalar data.

All arrays must be allocated by caller!

The data type of xk, fk, xi, fi and sens must be np.float64.

xk : in, (nk,3) array of neighbor point coordinates
fk : in, (nk,) array of function values at the neighbor points
xi : in, (3,) array, coordinates of the point xi
fi : in/out: if order=4, (35,) array containing (f, ...) at point xi (see wlsqm.fitter.defs)
             if order=3, (20,) array containing (f, ...) at point xi (see wlsqm.fitter.defs)
             if order=2, (10,) array containing (f, ...) at point xi (see wlsqm.fitter.defs)
             if order=1, (4,)  array containing (f, dfdx, dfdy, dfdz) at point xi
             if order=0, (1,)  array containing (f) at point xi
     on input:  those elements must be filled that correspond to the bitmask "knowns".
     on output: the unknown elements will be filled in (leaving the knowns untouched).
sens : out: if order=4, (nk,35) array containing sensitivity information.
            if order=3, (nk,20) array containing sensitivity information.
            if order=2, (nk,10) array containing sensitivity information.
            if order=1, (nk,4)  array containing sensitivity information.
            if order=0, (nk,1)  array containing sensitivity information.
            if fi[j] is unknown: sens[k,j] = d( fi[j] ) / d( fk[k] )
            if fi[j] is known:   sens[:,j] = nan (to indicate "not applicable").
do_sens : in, boolean: whether to perform sensitivity analysis. If False, "sens" can be none.
order  : in, order of the surrogate polynomial. Can be 0, 1, 2, 3 or 4.

         kth order fit gives derivatives up to order k, but usually (if the function is regular enough)
         the highest-order derivatives are practically nonsense, and increasing k increases the accuracy
         of the lower-order derivatives.

knowns : in, bitmask describing what is known about the function at the point xi.
         See the b3_* (bitmask, 3D case) constants in wlsqm.fitter.defs.

debug  : in, boolean. If debug is True, print row scale and condition number information to stdout.

weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
                   different weightings are good for different use cases of WLSQM.

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_3D_iterative())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic( 3, xk, None, fk, xi, 0., fi, sens, do_sens, order, knowns, weighting_method, debug )  # the None and 0. are dummies
    return iterations_taken

def fit_3D_iterative( double[::view.generic,::view.contiguous] xk, double[::view.generic] fk, double[::1] xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b3_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int max_iter=10, int debug=0 ):
    """Fit one local model to 3D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_3D(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative( 3, xk, None, fk, xi, 0., fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )  # the None and 0. are dummies
    return iterations_taken


def fit_3D_many( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int debug=0 ):
    """Fit many local models to 3D scalar data.

Each local model uses a different reference point xi, and different neighbor points xk.

Parameters almost like in fit_3D(), but with an extra leading axis for indexing the problem instance.

Note that each problem instance may have a different number of neighbor points, so nk is now a rank-1 array.
Other arrays having "k" as an index (below) must use  np.max(nk)  for the size of the axis that corresponds to k.

The nk array is used to determine which parts of the other arrays actually contain data; any unused elements (index >= nk[j]) will not be read.

For arrays, dtype np.float64 except where otherwise stated below.

xk:     [j,k,m] = problem_instance_index, neighbor_point_index, x_y_or_z
fk:     [j,k]
nk:     [j] (dtype np.int32)
xi:     [j,m]
fi:     [j,n] = problem_instance_index, DOF_number (of original unreduced system; see constants i?_* in wlsqm.fitter.defs)
sens:   [j,k,n]
order:  [j] (dtype np.int32)
knowns: [j] (dtype np.int64)
weighting_method: [j] (dtype np.int32)

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_3D_iterative_many())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many( 3, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, debug )
    return iterations_taken


def fit_3D_iterative_many( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int debug=0 ):
    """Fit many local models to 3D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_3D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many( 3, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )
    return iterations_taken


def fit_3D_many_parallel( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                          double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                          double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int ntasks=8, int debug=0 ):
    """Fit many local models to 3D scalar data; multi-threaded.

For parameters, see fit_3D_many(). In addition:

ntasks : in, number of threads for OpenMP

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_3D_iterative_many_parallel())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many_parallel( 3, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, ntasks, debug )
    return iterations_taken

def fit_3D_iterative_many_parallel( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int ntasks=8, int debug=0 ):
    """Fit many local models to 3D scalar data; multi-threaded. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_3D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.
ntasks   : in, number of threads for OpenMP

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many_parallel( 3, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, max_iter, ntasks, debug )
    return iterations_taken


##########
# 2D case
##########

def fit_2D( double[::view.generic,::view.contiguous] xk, double[::view.generic] fk, double[::1] xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b2_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int debug=0 ):
    """Fit one local model to 2D scalar data.

All arrays must be allocated by caller!

The data type of xk, fk, xi, fi and sens must be np.float64.

xk : in, (nk,2) array of neighbor point coordinates
fk : in, (nk,) array of function values at the neighbor points
xi : in, (2,) array, coordinates of the point xi
fi : in/out: if order=4, (15,) array containing (f, dfdx, dfdy, d2fdx2, d2fdxdy, d2fdy2, d3fdx3, d3fdx2dy, d3fdxdy2, d3fdy3, d4fdx4, d4fdx3dy, d4fdx2dy2, d4fdxdy3, d4fdy4) at point xi
             if order=3, (10,) array containing (f, dfdx, dfdy, d2fdx2, d2fdxdy, d2fdy2, d3fdx3, d3fdx2dy, d3fdxdy2, d3fdy3) at point xi
             if order=2, (6,)  array containing (f, dfdx, dfdy, d2fdx2, d2fdxdy, d2fdy2) at point xi
             if order=1, (3,)  array containing (f, dfdx, dfdy) at point xi
             if order=0, (1,)  array containing (f) at point xi
     on input:  those elements must be filled that correspond to the bitmask "knowns".
     on output: the unknown elements will be filled in (leaving the knowns untouched).
sens : out: if order=4, (nk,15) array containing sensitivity information.
            if order=3, (nk,10) array containing sensitivity information.
            if order=2, (nk,6)  array containing sensitivity information.
            if order=1, (nk,3)  array containing sensitivity information.
            if order=0, (nk,1)  array containing sensitivity information.
            if fi[j] is unknown: sens[k,j] = d( fi[j] ) / d( fk[k] )
            if fi[j] is known:   sens[:,j] = nan (to indicate "not applicable").
do_sens : in, boolean: whether to perform sensitivity analysis. If False, "sens" can be none.
order  : in, order of the surrogate polynomial. Can be 0, 1, 2, 3 or 4.

         kth order fit gives derivatives up to order k, but usually (if the function is regular enough)
         the highest-order derivatives are practically nonsense, and increasing k increases the accuracy
         of the lower-order derivatives.

knowns : in, bitmask describing what is known about the function at the point xi.
         See the b2_* (bitmask, 2D case) constants.

debug  : in, boolean. If debug is True, print row scale and condition number information to stdout.

weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
                   different weightings are good for different use cases of WLSQM.

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_2D_iterative())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic( 2, xk, None, fk, xi, 0., fi, sens, do_sens, order, knowns, weighting_method, debug )  # the None and 0. are dummies
    return iterations_taken


def fit_2D_iterative( double[::view.generic,::view.contiguous] xk, double[::view.generic] fk, double[::1] xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b2_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int max_iter=10, int debug=0 ):
    """Fit one local model to 2D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_2D(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative( 2, xk, None, fk, xi, 0., fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )  # the None and 0. are dummies
    return iterations_taken


# Basic algorithm, many problem instances.
#
# Parameters almost like in fit_2D(), but with an extra leading dimension for indexing the problem instance.
#
#
def fit_2D_many( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int debug=0 ):
    """Fit many local models to 2D scalar data.

Each local model uses a different reference point xi, and different neighbor points xk.

Parameters almost like in fit_2D(), but with an extra leading axis for indexing the problem instance.

Note that each problem instance may have a different number of neighbor points, so nk is now a rank-1 array.
Other arrays having "k" as an index (below) must use  np.max(nk)  for the size of the axis that corresponds to k.

The nk array is used to determine which parts of the other arrays actually contain data; any unused elements (index >= nk[j]) will not be read.

For arrays, dtype np.float64 except where otherwise stated below.

xk:     [j,k,m] = problem_instance_index, neighbor_point_index, x_or_y
fk:     [j,k]
nk:     [j] (dtype np.int32)
xi:     [j,m]
fi:     [j,n] = problem_instance_index, DOF_number (of original unreduced system; see constants i?_* in wlsqm.fitter.defs)
sens:   [j,k,n]
order:  [j] (dtype np.int32)
knowns: [j] (dtype np.int64)
weighting_method: [j]

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_2D_iterative_many())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many( 2, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, debug )
    return iterations_taken


def fit_2D_iterative_many( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int debug=0 ):
    """Fit many local models to 2D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_2D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many( 2, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )
    return iterations_taken


def fit_2D_many_parallel( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                          double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                          double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int ntasks=8, int debug=0 ):
    """Fit many local models to 2D scalar data; multi-threaded.

For parameters, see fit_2D_many(). In addition:

ntasks : in, number of threads for OpenMP

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_2D_iterative_many_parallel())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many_parallel( 2, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, ntasks, debug )
    return iterations_taken


def fit_2D_iterative_many_parallel( double[::view.generic,::view.generic,::view.contiguous] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic,::view.contiguous] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int ntasks=8, int debug=0 ):
    """Fit many local models to 2D scalar data; multi-threaded. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_2D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.
ntasks   : in, number of threads for OpenMP

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many_parallel( 2, xk, None, fk, nk, xi, None, fi, sens, do_sens, order, knowns, weighting_method, max_iter, ntasks, debug )
    return iterations_taken


##########
# 1D case
##########

# Basic algorithm, one problem instance.
def fit_1D( double[::view.generic] xk, double[::view.generic] fk, double xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b1_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int debug=0 ):
    """Fit one local model to 1D scalar data.

All arrays must be allocated by caller!

The data type of xk, fk, xi, fi and sens must be np.float64.

xk : in, (nk,) array of neighbor point coordinates
fk : in, (nk,) array of function values at the neighbor points
xi : in, double, coordinate of the point xi
fi : in/out: if order=4, (5,) array containing (f, dfdx, d2fdx2, d3fdx3, d4fdx4) at point xi
             if order=3, (4,) array containing (f, dfdx, d2fdx2, d3fdx3) at point xi
             if order=2, (3,) array containing (f, dfdx, d2fdx2) at point xi
             if order=1, (2,) array containing (f, dfdx) at point xi
             if order=0, (1,) array containing (f) at point xi
     on input:  those elements must be filled that correspond to the bitmask "knowns".
     on output: the unknown elements will be filled in (leaving the knowns untouched).
sens : out: if order=4, (nk,5) array containing sensitivity information.
            if order=3, (nk,4) array containing sensitivity information.
            if order=2, (nk,3) array containing sensitivity information.
            if order=1, (nk,2) array containing sensitivity information.
            if order=0, (nk,1) array containing sensitivity information.
            if fi[j] is unknown: sens[k,j] = d( fi[j] ) / d( fk[k] )
            if fi[j] is known:   sens[:,j] = nan (to indicate "not applicable").
do_sens : in, boolean: whether to perform sensitivity analysis. If False, "sens" can be none.
order  : in, order of the surrogate polynomial. Can be 0, 1, 2, 3 or 4.

         kth order fit gives derivatives up to order k, but usually (if the function is regular enough)
         the highest-order derivatives are practically nonsense, and increasing k increases the accuracy
         of the lower-order derivatives.

knowns : in, bitmask describing what is known about the function at the point xi.
         See the b1_* (bitmask, 1D case) constants.

debug  : in, boolean. If debug is True, print row scale and condition number information to stdout.

weighting_method : in, one of the constants WEIGHT_*. Specifies the type of weighting to use;
                   different weightings are good for different use cases of WLSQM.

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_1D_iterative())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic( 1, None, xk, fk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, debug )  # the Nones are dummies
    return iterations_taken


def fit_1D_iterative( double[::view.generic] xk, double[::view.generic] fk, double xi, double[::1] fi,
            double[::view.generic,::view.contiguous] sens, int do_sens=0, int order=2, long long knowns=defs.b1_F_c, int weighting_method=defs.WEIGHT_CENTER_c, int max_iter=10, int debug=0 ):
    """Fit one local model to 1D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_1D(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative( 1, None, xk, fk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )  # the Nones are dummies
    return iterations_taken


def fit_1D_many( double[::view.generic,::view.generic] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int debug=0 ):
    """Fit many local models to 1D scalar data.

Each local model uses a different reference point xi, and different neighbor points xk.

Parameters almost like in fit_1D(), but with an extra leading axis for indexing the problem instance.

Note that each problem instance may have a different number of neighbor points, so nk is now a rank-1 array.
Other arrays having "k" as an index (below) must use  np.max(nk)  for the size of the axis that corresponds to k.

The nk array is used to determine which parts of the other arrays actually contain data; any unused elements (index >= nk[j]) will not be read.

For arrays, dtype np.float64 except where otherwise stated below.

xk:     [j,k] = problem_instance_index, neighbor_point_index
fk:     [j,k]
nk:     [j] (dtype np.int32)
xi:     [j]
fi:     [j,n] = problem_instance_index, DOF_number (of original unreduced system; see constants i?_* in wlsqm.fitter.defs)
sens:   [j,k,n]
order:  [j] (dtype np.int32)
knowns: [j] (dtype np.int64)
weighting_method: [j]

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_1D_iterative_many())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many( 1, None, xk, fk, nk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, debug )
    return iterations_taken


def fit_1D_iterative_many( double[::view.generic,::view.generic] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int debug=0 ):
    """Fit many local models to 1D scalar data. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_1D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many( 1, None, xk, fk, nk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, max_iter, debug )
    return iterations_taken


def fit_1D_many_parallel( double[::view.generic,::view.generic] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int ntasks=8, int debug=0 ):
    """Fit many local models to 1D scalar data; multi-threaded.

For parameters, see fit_1D_many(). In addition:

ntasks : in, number of threads for OpenMP

Return value:
    always 0 (no corrective iterations; this is for consistency with fit_1D_iterative_many_parallel())
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_basic_many_parallel( 1, None, xk, fk, nk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, ntasks, debug )
    return iterations_taken


def fit_1D_iterative_many_parallel( double[::view.generic,::view.generic] xk, double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                 double[::view.generic] xi, double[::view.generic,::view.contiguous] fi,
                 double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens, int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter=10, int ntasks=8, int debug=0 ):
    """Fit many local models to 1D scalar data; multi-threaded. Algorithm with iterative refinement to mitigate roundoff.

For parameters, see fit_1D_many(). In addition:

max_iter : in, maximum number of fitting iterations to take.
ntasks   : in, number of threads for OpenMP

Return value:
    number of corrective iterations actually taken. May be < max_iter, if convergence was reached earlier.
"""
    cdef int iterations_taken
    with nogil:
        iterations_taken = generic_fit_iterative_many_parallel( 1, None, xk, fk, nk, None, xi, fi, sens, do_sens, order, knowns, weighting_method, max_iter, ntasks, debug )
    return iterations_taken


####################################################
# C API
####################################################

####################################################
# Single case (one neighborhood), single-threaded
####################################################

# Basic algorithm, one problem instance.
#
# We take in parameters for both 1D and ManyD (2D and 3D) cases; only one set is actually used depending on dimension.
# This is used to work around the static typing in C to allow a single dimension-agnostic implementation for fitting.
#
cdef int generic_fit_basic( int dimension, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D, double[::view.generic] fk, double[::1] xiManyD, double xi1D, double[::1] fi,
                                           double[::view.generic,::view.contiguous] sens, int do_sens, int order, long long knowns, int weighting_method, int debug ) nogil except -1:

    DEF TASKID = 0  # no parallel processing, so we have only one task

    # Create the Case, allocating memory and initializing the DOF mapping arrays.
    #
    # Unused components can be initialized to 0, their values are not used.
    #
    cdef double xi, yi, zi
    if dimension == 1:
        xi = xi1D
        yi = 0.
        zi = 0.
    else:
        xi = xiManyD[0]
        yi = xiManyD[1]
        zi = xiManyD[2]  if  dimension == 3  else 0.
    cdef int nk = fk.shape[0]  # number of neighbors (data points used in fit)
    cdef infra.Case* case = infra.Case_new( dimension, order, xi, yi, zi, nk, knowns, weighting_method, do_sens, iterative=0, manager=<infra.CaseManager*>0, host=<infra.Case*>0 )

    # Create distance matrix and weights from xk and xi.
    #
    impl.make_c_nD( case, xkManyD, xk1D )  # only either *ManyD or *1D are used, depending on dimension

    # using the distance data in c and the weights in w, create the reduced A
    impl.make_A( case )

    # precondition and factorize A
    impl.preprocess_A( case, debug )

    # use the preprocessed A to perform the fit
    cdef double* my_fi = &fi[0]
    infra.Case_set_fi( case, my_fi )  # populate knowns
    impl.solve( case, fk, sens, do_sens, TASKID )
    infra.Case_get_fi( case, my_fi )  # write solution into user-given array fi

    # Destroy the Case.
    #
    infra.Case_del( case )

    return 0  # number of refinement iterations taken; always 0 for this algorithm


# Algorithm with iterative refinement to reduce numerical error, one problem instance.
#
# We take in parameters for both 1D and ManyD (2D and 3D) cases; only one set is actually used depending on dimension.
# This is used to work around the static typing in C to allow a single dimension-agnostic implementation for fitting.
#
cdef int generic_fit_iterative( int dimension, double[::view.generic,::view.contiguous] xkManyD, double[::view.generic] xk1D, double[::view.generic] fk, double[::1] xiManyD, double xi1D, double[::1] fi,
                                               double[::view.generic,::view.contiguous] sens, int do_sens, int order, long long knowns, int weighting_method, int max_iter, int debug ) nogil except -1:

    DEF TASKID = 0  # no parallel processing, so we have only one task

    cdef double xi, yi, zi
    if dimension == 1:
        xi = xi1D
        yi = 0.
        zi = 0.
    else:
        xi = xiManyD[0]
        yi = xiManyD[1]
        zi = xiManyD[2]  if  dimension == 3  else 0.

    cdef int nk = fk.shape[0]  # number of neighbors (data points used in fit)
    cdef infra.Case* case = infra.Case_new( dimension, order, xi, yi, zi, nk, knowns, weighting_method, do_sens, iterative=1, manager=<infra.CaseManager*>0, host=<infra.Case*>0 )

    impl.make_c_nD( case, xkManyD, xk1D )  # only either *ManyD or *1D are used, depending on dimension
    impl.make_A( case )
    impl.preprocess_A( case, debug )

    cdef double* my_fi = &fi[0]
    infra.Case_set_fi( case, my_fi )  # populate knowns
    cdef int iterations_taken = impl.solve_iterative( case, fk, sens, do_sens, TASKID, max_iter, xkManyD, xk1D )
    infra.Case_get_fi( case, my_fi )  # write solution into user-given array fi

    infra.Case_del( case )

    return iterations_taken


####################################################
# Many cases, single-threaded
####################################################

# Basic algorithm, many problem instances.
#
# Strategy for multiple problem instances (with the same dimension,order,knowns,do_sens,iterative):
#  - create a CaseManager instance
#  - create the Case instances, passing in the CaseManager instance to use centralized mode for memory allocation
#    - this will automatically call CaseManager_add() for each, saving the pointers to the Case instances into the manager
#  - when all cases have been added to the manager, call CaseManager_commit()
#    - this allocates a single space-optimal buffer that will accommodate all the cases
#    - this also calls Case_allocate() for each managed Case instance
#  - then just loop over the Case objects stored in the CaseManager (in parallel), solving each case.
#    - use Case_get_wrk(), Case_get_fk_tmp(), Case_get_fi_tmp() to get access to per-task (not per-problem-instance) buffers; these buffers live in the manager.
#  - in the driver version, destroy the CaseManager object when finished
#    - this will also destroy the managed Case objects
#
# xkManyD: [j,k,m] = case, neighbor point, x_y_or_z
# xk1D:    [j,k]
# fk:      [j,k]
# nk:      [j]
# xiManyD: [j,m]
# xi1D:    [j]
# fi:      [j,n] = case, DOF_number (of original unreduced system; see constants i?_* in wlsqm.fitter.defs)
# sens:    [j,k,n]
# order:   [j]
# knowns:  [j]
# weighting_method: [j]
#
# The x_or_y and DOF_number dimensions are contiguous, the rest have generic memory layout.
#
cdef int generic_fit_basic_many( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int debug ) nogil except -1:

    DEF NTASKS = 1
    DEF TASKID = 0     # no parallel processing, so we have only one task
    DEF ITERATIVE = 0  # basic algorithm, not iterative

    # Create the CaseManager to handle centralized memory allocation.
    #
    cdef int ncases = nk.shape[0]
    cdef infra.CaseManager* manager = infra.CaseManager_new( dimension, do_sens, ITERATIVE, ncases, NTASKS )

    # We have to explicitly dummy out the parameters for the "wrong" number of space dimensions, since we may have gotten
    # dummy parameters ourselves and hence cannot index into them. Thus we need to switch on dimension here.
    # We'll do it outside the loop for a total extra cost of one instruction, independent of the number of problem instances (ncases).
    #
    cdef infra.Case* case
    cdef double* my_fi
    cdef double xi, yi, zi
    cdef int j, nkj
    if dimension >= 2:
        # Create the Cases, adding them to the manager.
        #
        if dimension == 3:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                zi = xiManyD[j,2]
                # this will automatically add the case to the manager, so we don't need to save the returned pointer
                infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        else: # dimension == 2:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                infra.Case_new( dimension, order[j], xi, yi, 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # Solve the cases. This updates the internal fi arrays of the Case instances.
        #
        for j in range(ncases):
            case  = manager.cases[j]
            nkj   = nk[j]  # each case has its own value of nk; this determines the part of the xk and fk arrays that is actually used
            my_fi = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, xkManyD[j,:nkj,:], None )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )  # populate knowns (in managed mode, must come after CaseManager_commit())
            if sens is None:  # must check since None cannot be sliced
                impl.solve( case, fk[j,:nkj], None, do_sens, TASKID )
            else:
                impl.solve( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID )

        # Write the solution into user-given fi array.
        #
        # We must do this after the solve loop has finished, to ensure that all cases get equal treatment for their input data.
        #
        # (It is possible that fk and the user-given fi are actually views into the same physical array,
        #  and that the solve updates the function value column of fi, where fk gets its data from.
        #  Thus, to be sure, we update the user-given fi only after all cases have been solved.)
        #
        for j in range(ncases):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    else: # dimension == 1:
        # Create the Cases, adding them to the manager.
        #
        for j in range(ncases):
            xi = xi1D[j]
            infra.Case_new( dimension, order[j], xi, 0., 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # Solve the cases. This updates the internal fi arrays of the Case instances.
        #
        for j in range(ncases):
            case  = manager.cases[j]
            nkj   = nk[j]
            my_fi = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, None, xk1D[j,:nkj] )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                impl.solve( case, fk[j,:nkj], None, do_sens, TASKID )
            else:
                impl.solve( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID )

        # Write the solution into user-given fi array.
        #
        # We must do this after the solve loop has finished, to ensure that all cases get equal treatment for their input data.
        #
        # (It is possible that fk and the user-given fi are actually views into the same physical array,
        #  and that the solve updates the function value column of fi, where fk gets its data from.
        #  Thus, to be sure, we update the user-given fi only after all cases have been solved.)
        #
        for j in range(ncases):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    infra.CaseManager_del( manager )  # this will also destroy the managed Case instances

    return 0  # number of refinement iterations taken; always 0 for this algorithm


# Algorithm with iterative refinement, many problem instances.
#
# xkManyD: [j,k,m] = case, neighbor point, x_y_or_z
# xk1D:    [j,k]
# fk:      [j,k]
# nk:      [j]
# xiManyD: [j,m]
# xi1D:    [j]
# fi:      [j,n] = case, DOF_number (of original unreduced system; see constants i?_* in wlsqm.fitter.defs)
# sens:    [j,k,n]
# order:   [j]
# knowns:  [j]
# weighting_method: [j]
#
# The x_or_y and DOF_number dimensions are contiguous, the rest have generic memory layout.
#
cdef int generic_fit_iterative_many( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                    double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                    double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                    double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                    int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter, int debug ) nogil except -1:

    DEF NTASKS = 1
    DEF TASKID = 0     # no parallel processing, so we have only one task
    DEF ITERATIVE = 1  # iterative algorithm

    cdef int max_iterations_taken = 0

    # Create the CaseManager to handle centralized memory allocation.
    #
    cdef int ncases = nk.shape[0]
    cdef infra.CaseManager* manager = infra.CaseManager_new( dimension, do_sens, ITERATIVE, ncases, NTASKS )

    # Create the Cases, adding them to the manager.
    #
    cdef infra.Case* case
    cdef double* my_fi
    cdef double xi, yi, zi
    cdef int iterations_taken
    cdef int j, nkj
    if dimension >= 2:
        # add to manager
        #
        if dimension == 3:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                zi = xiManyD[j,2]
                # this will automatically add the case to the manager, so we don't need to save the returned pointer
                infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        else: # dimension == 2:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                infra.Case_new( dimension, order[j], xi, yi, 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # solve
        #
        for j in range(ncases):
            case  = manager.cases[j]
            nkj   = nk[j]  # each case has its own value of nk; this determines the part of the xk and fk arrays that is actually used
            my_fi = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, xkManyD[j,:nkj,:], None )  # explicitly dummy out the 1D parameters, since we may have gotten dummy parameters ourselves and hence cannot index into them
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], None, do_sens, TASKID, max_iter, xkManyD[j,:nkj,:], None )
            else:
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID, max_iter, xkManyD[j,:nkj,:], None )
            if iterations_taken > max_iterations_taken:
                max_iterations_taken = iterations_taken

        # get solution
        #
        for j in range(ncases):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    else: # dimension == 1:
        # add to manager
        #
        for j in range(ncases):
            xi = xi1D[j]
            infra.Case_new( dimension, order[j], xi, 0., 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )

        # solve
        #
        for j in range(ncases):
            case  = manager.cases[j]
            nkj   = nk[j]
            my_fi = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, None, xk1D[j,:nkj] )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], None, do_sens, TASKID, max_iter, None, xk1D[j,:nkj] )
            else:
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID, max_iter, None, xk1D[j,:nkj] )
            if iterations_taken > max_iterations_taken:
                max_iterations_taken = iterations_taken

        # get solution
        #
        for j in range(ncases):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    infra.CaseManager_del( manager )

    return max_iterations_taken


####################################################
# Many cases, multithreaded
####################################################

# ntasks : in, number of threads to use for computation
#
# otherwise like generic_fit_basic_many()
#
cdef int generic_fit_basic_many_parallel( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                         double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                         double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                         double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                         int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int ntasks, int debug ) nogil except -1:

    DEF ITERATIVE = 0  # basic algorithm, not iterative

    # Create the CaseManager to handle centralized memory allocation.
    #
    cdef int ncases = nk.shape[0]
    cdef infra.CaseManager* manager = infra.CaseManager_new( dimension, do_sens, ITERATIVE, ncases, ntasks )

    # We have to explicitly dummy out the parameters for the "wrong" number of space dimensions, since we may have gotten
    # dummy parameters ourselves and hence cannot index into them. Thus we need to switch on dimension here.
    # We'll do it outside the loop for a total extra cost of one instruction, independent of the number of problem instances (ncases).
    #
    cdef infra.Case* case
    cdef double* my_fi
    cdef double xi, yi, zi
    cdef int j, nkj, taskid
    if dimension >= 2:
        # Create the Cases, adding them to the manager.
        #
        # The initialization is not thread-safe, because Case_new() calls CaseManager_add(), which updates its array of managed cases.
        # Because the array would need to be locked for thread-safety, it is much simpler (and also probably faster) to just do this in one thread,
        # since all the constructor does is to update a few member variables before calling CaseManager_add().
        #
        if dimension == 3:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                zi = xiManyD[j,2]
                # this will automatically add the case to the manager, so we don't need to save the returned pointer
                infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        else: # dimension == 2:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                infra.Case_new( dimension, order[j], xi, yi, 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # Solve the cases. This updates the internal fi arrays of the Case instances.
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            taskid = openmp.omp_get_thread_num()
            case   = manager.cases[j]
            nkj    = nk[j]  # each case has its own value of nk; this determines the part of the xk and fk arrays that is actually used
            my_fi  = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, xkManyD[j,:nkj,:], None )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )  # populate knowns (in managed mode, must come after CaseManager_commit())
            if sens is None:  # must check since None cannot be sliced
                impl.solve( case, fk[j,:nkj], None, do_sens, taskid )
            else:
                impl.solve( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, taskid )

        # Write the solution into user-given fi array.
        #
        # We must do this after the solve loop has finished, to ensure that all cases get equal treatment for their input data.
        #
        # (It is possible that fk and the user-given fi are actually views into the same physical array,
        #  and that the solve updates the function value column of fi, where fk gets its data from.
        #  Thus, to be sure, we update the user-given fi only after all cases have been solved.)
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    else: # dimension == 1:
        # Create the Cases, adding them to the manager.
        #
        for j in range(ncases):
            xi = xi1D[j]
            infra.Case_new( dimension, order[j], xi, 0., 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # Solve the cases. This updates the internal fi arrays of the Case instances.
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            taskid = openmp.omp_get_thread_num()
            case   = manager.cases[j]
            nkj    = nk[j]
            my_fi  = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, None, xk1D[j,:nkj] )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                impl.solve( case, fk[j,:nkj], None, do_sens, TASKID )
            else:
                impl.solve( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, TASKID )

        # Write the solution into user-given fi array.
        #
        # We must do this after the solve loop has finished, to ensure that all cases get equal treatment for their input data.
        #
        # (It is possible that fk and the user-given fi are actually views into the same physical array,
        #  and that the solve updates the function value column of fi, where fk gets its data from.
        #  Thus, to be sure, we update the user-given fi only after all cases have been solved.)
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    infra.CaseManager_del( manager )  # this will also destroy the managed Case instances

    return 0  # number of refinement iterations taken; always 0 for this algorithm


# ntasks : in, number of threads to use for computation
#
# otherwise like generic_fit_iterative_many()
#
cdef int generic_fit_iterative_many_parallel( int dimension, double[::view.generic,::view.generic,::view.contiguous] xkManyD, double[::view.generic,::view.generic] xk1D,
                                                             double[::view.generic,::view.generic] fk, int[::view.generic] nk,
                                                             double[::view.generic,::view.contiguous] xiManyD, double[::view.generic] xi1D, double[::view.generic,::view.contiguous] fi,
                                                             double[::view.generic,::view.generic,::view.contiguous] sens, int do_sens,
                                                             int[::view.generic] order, long long[::view.generic] knowns, int[::view.generic] weighting_method, int max_iter, int ntasks, int debug ) nogil except -1:

    DEF ITERATIVE = 1  # iterative algorithm

    # we need an ntasks-sized array to find max_iterations_taken, since the solving now proceeds in parallel
    cdef int* max_iterations_taken = <int*>malloc( ntasks*sizeof(int) )
    cdef int taskid
    for taskid in range(ntasks):
        max_iterations_taken[taskid] = 0

    # Create the CaseManager to handle centralized memory allocation.
    #
    cdef int ncases = nk.shape[0]
    cdef infra.CaseManager* manager = infra.CaseManager_new( dimension, do_sens, ITERATIVE, ncases, ntasks )

    # Create the Cases, adding them to the manager.
    #
    cdef infra.Case* case
    cdef double* my_fi
    cdef double xi, yi, zi
    cdef int iterations_taken
    cdef int j, nkj
    if dimension >= 2:
        # add to manager
        #
        if dimension == 3:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                zi = xiManyD[j,2]
                # this will automatically add the case to the manager, so we don't need to save the returned pointer
                infra.Case_new( dimension, order[j], xi, yi, zi, nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        else: # dimension == 2:
            for j in range(ncases):
                xi = xiManyD[j,0]
                yi = xiManyD[j,1]
                infra.Case_new( dimension, order[j], xi, yi, 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )  # done adding cases

        # solve
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            taskid = openmp.omp_get_thread_num()
            case   = manager.cases[j]
            nkj    = nk[j]  # each case has its own value of nk; this determines the part of the xk and fk arrays that is actually used
            my_fi  = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, xkManyD[j,:nkj,:], None )  # explicitly dummy out the 1D parameters, since we may have gotten dummy parameters ourselves and hence cannot index into them
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], None, do_sens, taskid, max_iter, xkManyD[j,:nkj,:], None )
            else:
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, taskid, max_iter, xkManyD[j,:nkj,:], None )
            if iterations_taken > max_iterations_taken[taskid]:
                max_iterations_taken[taskid] = iterations_taken

        # get solution
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    else: # dimension == 1:
        # add to manager
        #
        for j in range(ncases):
            xi = xi1D[j]
            infra.Case_new( dimension, order[j], xi, 0., 0., nk[j], knowns[j], weighting_method[j], do_sens, ITERATIVE, manager, host=<infra.Case*>0 )
        infra.CaseManager_commit( manager )

        # solve
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            taskid = openmp.omp_get_thread_num()
            case   = manager.cases[j]
            nkj    = nk[j]
            my_fi  = &fi[j,0]  # user array for coefficient data for the jth case
            impl.make_c_nD( case, None, xk1D[j,:nkj] )
            impl.make_A( case )
            impl.preprocess_A( case, debug )
            infra.Case_set_fi( case, my_fi )
            if sens is None:  # must check since None cannot be sliced
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], None, do_sens, taskid, max_iter, None, xk1D[j,:nkj] )
            else:
                iterations_taken = impl.solve_iterative( case, fk[j,:nkj], sens[j,:nkj,:], do_sens, taskid, max_iter, None, xk1D[j,:nkj] )
            if iterations_taken > max_iterations_taken[taskid]:
                max_iterations_taken[taskid] = iterations_taken

        # get solution
        #
        for j in cython.parallel.prange(ncases, num_threads=ntasks):
            infra.Case_get_fi( manager.cases[j], &fi[j,0] )

    infra.CaseManager_del( manager )

    # get the maximum number of iterations taken in any thread
    cdef int total_max_iterations_taken=0
    for taskid in range(ntasks):
        if max_iterations_taken[taskid] > total_max_iterations_taken:
            total_max_iterations_taken = max_iterations_taken[taskid]
    free( <void*>max_iterations_taken )

    return total_max_iterations_taken

