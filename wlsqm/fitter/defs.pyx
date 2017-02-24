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
#
"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

This module contains C-level and Python-level definitions of constants. The constants are made visible to Python by creating Python objects, with their values copied from the corresponding C constants.

In the source code, the suffix of _c means "visible at the C level in Cython"; it is used to distinguish the typed C constants from the corresponding Python objects.

Naming scheme:

    ALGO_*   = algorithms for the solve step (for advanced API in wlsqm.fitter.expert).
    WEIGHT_* = weighting methods for the error (data - predicted) in least-squares fitting.

    i1_* = integer, 1D case
    i2_* = integer, 2D case
    i3_* = integer, 3D case

        The i?_* constants are the human-readable names for the DOF indices in the "fi" array (see docstrings in wlsqm.fitter.simple).

    b1_* = bitmask, 1D case
    b2_* = bitmask, 2D case
    b3_* = bitmask, 3D case

    *_end = one-past-end index of this case. The ordinal (0th, 1st, 2nd, 3rd, 4th) refers to the degree of the fit.
            E.g. i2_3rd_end = one-past-end for 2D with 3rd order fit.

    SIZE1 = maximum possible number of DOFs (degrees of freedom), 1D case
    SIZE2 = maximum possible number of DOFs (degrees of freedom), 2D case
    SIZE3 = maximum possible number of DOFs (degrees of freedom), 3D case

        ("maximum possible" because if order < 4, then only lower-degree DOFs will exist.)

    F  = function value
    X  = "times x" (coefficients) or "differentiate by x" (see wlsqm.fitter.interp)
    X2 = "times x**2" or "differentiate twice by x"
    Y, Z respectively

Examples:

    i2_F   = 2D case, function value
    i2_X2Y = 2D case, coefficient of the X**2 * Y term in the polynomial; or request differentiation twice by x and once by y to compute d3f/dx2dy (see wlsqm.fitter.interp)

        IMPORTANT: the DOF values returned by the fitter are "partially baked" such that the DOF value directly corresponds to the value of the corresponding derivative.
                   This is for convenience of evaluating derivatives at the model reference point.

                   E.g. fi[:,i2_X2] is the coefficient of d2f/dx2 in a Taylor series expansion of f around the reference point xi.
                   (The ":" is here meant to refer to the reference point xi for all local models; see  wlsqm.fitter.simple.fit_2D_many()  for a description of the "fi" array.)

JJ 2016-11-30
"""

#################################################
# C definitions (Cython level)
#################################################

# Algorithms for the solve step (expert mode).
#
cdef int ALGO_BASIC_c     = 1  # just fit once
cdef int ALGO_ITERATIVE_c = 2  # fit with iterative refinement to mitigate roundoff

# Weighting methods.
#
cdef int WEIGHT_UNIFORM_c = 1
cdef int WEIGHT_CENTER_c  = 2

# DOF index in the array f.
#
# These are ordered in increasing order of number of differentiations, so that if only first derivatives
# are required, the DOF array can be simply truncated after the first derivatives.
#
# To avoid gaps in the numbering, this requires separate DOF orderings for the 1D, 2D and 3D cases.
#
# (The other logical possibility would be function value first, then x-related, then y-related, then mixed,
#  but then the case of "first derivatives only" requires changes to the ordering to avoid gaps.
#  Specifying different orderings for different numbers of space dimensions is conceptually cleaner
#  of the two possibilities.)

# 1D case
#
cdef int i1_F_c   = 0
cdef int i1_X_c   = 1
cdef int i1_X2_c  = 2
cdef int i1_X3_c  = 3
cdef int i1_X4_c  = 4

cdef int i1_0th_end_c = 1  # one-past end of zeroth-order case
cdef int i1_1st_end_c = 2  # one-past end of first-order case
cdef int i1_2nd_end_c = 3  # one-past-end of second-order case
cdef int i1_3rd_end_c = 4  # one-past-end of third-order case
cdef int i1_4th_end_c = 5  # one-past-end of fourth-order case

cdef int SIZE1_c = i1_4th_end_c  # maximum possible number of DOFs, 1D case

# 2D case
#
cdef int i2_F_c     =  0

cdef int i2_X_c     =  1
cdef int i2_Y_c     =  2

cdef int i2_X2_c    =  3
cdef int i2_XY_c    =  4
cdef int i2_Y2_c    =  5

cdef int i2_X3_c    =  6
cdef int i2_X2Y_c   =  7
cdef int i2_XY2_c   =  8
cdef int i2_Y3_c    =  9

cdef int i2_X4_c    = 10
cdef int i2_X3Y_c   = 11
cdef int i2_X2Y2_c  = 12
cdef int i2_XY3_c   = 13
cdef int i2_Y4_c    = 14

cdef int i2_0th_end_c =  1  # one-past end of zeroth-order case
cdef int i2_1st_end_c =  3  # one-past end of first-order case
cdef int i2_2nd_end_c =  6  # one-past-end of second-order case
cdef int i2_3rd_end_c = 10  # one-past-end of third-order case
cdef int i2_4th_end_c = 15  # one-past-end of fourth-order case

cdef int SIZE2_c = i2_4th_end_c  # maximum possible number of DOFs, 2D case

# 3D case
#
cdef int i3_F_c      =  0

cdef int i3_X_c      =  1
cdef int i3_Y_c      =  2
cdef int i3_Z_c      =  3

cdef int i3_X2_c     =  4
cdef int i3_XY_c     =  5
cdef int i3_Y2_c     =  6
cdef int i3_YZ_c     =  7
cdef int i3_Z2_c     =  8
cdef int i3_XZ_c     =  9

cdef int i3_X3_c     = 10
cdef int i3_X2Y_c    = 11
cdef int i3_XY2_c    = 12
cdef int i3_Y3_c     = 13
cdef int i3_Y2Z_c    = 14
cdef int i3_YZ2_c    = 15
cdef int i3_Z3_c     = 16
cdef int i3_XZ2_c    = 17
cdef int i3_X2Z_c    = 18
cdef int i3_XYZ_c    = 19

cdef int i3_X4_c     = 20
cdef int i3_X3Y_c    = 21
cdef int i3_X2Y2_c   = 22
cdef int i3_XY3_c    = 23
cdef int i3_Y4_c     = 24
cdef int i3_Y3Z_c    = 25
cdef int i3_Y2Z2_c   = 26
cdef int i3_YZ3_c    = 27
cdef int i3_Z4_c     = 28
cdef int i3_XZ3_c    = 29
cdef int i3_X2Z2_c   = 30
cdef int i3_X3Z_c    = 31
cdef int i3_X2YZ_c   = 32
cdef int i3_XY2Z_c   = 33
cdef int i3_XYZ2_c   = 34

cdef int i3_0th_end_c =  1  # one-past end of zeroth-order case
cdef int i3_1st_end_c =  4  # one-past end of first-order case
cdef int i3_2nd_end_c = 10  # one-past-end of second-order case
cdef int i3_3rd_end_c = 20  # one-past-end of third-order case
cdef int i3_4th_end_c = 35  # one-past-end of fourth-order case

cdef int SIZE3_c = i3_4th_end_c  # maximum possible number of DOFs, 3D case


# bitmask constants for knowns.
#
# Knowns are eliminated algebraically from the equation system; if any knowns are specified,
# the system to be solved (for a point x_i) will be smaller than the full system (e.g. "full" is 6x6 for 2nd order in 2D).
#
# The sensible default is to consider the function value F known, with all the derivatives unknown.
#
# Note that here "known" means "known at point xi" (the reference point of the model).
#
# Function values (F) are always assumed known at all *neighbor* points xk, since they are used
# for determining the local least-squares polynomial fit to the data. This fit is then used
# as a local surrogate model representing the unknown function f.
#
# In the application context of solving IBVPs with explicit time integration, the option to have the function value (F) as an unknown
# is useful with Neumann BCs. The neighborhoods of the Neumann boundary points can be chosen such that each Neumann boundary point
# only uses neighbors from the interior of the domain. This gives the possibility to leave F free at all Neumann boundary points,
# while prescribing only a derivative (the normal-direction derivative).
#
# (In practice, at slanted (i.e. not coordinate axis aligned) boundaries, local (tangent, normal)
#  coordinates must be used; i.e., the coordinate system in which the derivatives are to be computed
#  must be rotated to match the orientation of the boundary. This makes Y the normal derivative,
#  which can then be prescribed using this mechanism, while leaving the function value F free.)

# 1D case
#
cdef long long b1_F_c      = (1LL << i1_F_c)
cdef long long b1_X_c      = (1LL << i1_X_c)
cdef long long b1_X2_c     = (1LL << i1_X2_c)
cdef long long b1_X3_c     = (1LL << i1_X3_c)
cdef long long b1_X4_c     = (1LL << i1_X4_c)

# 2D case
#
cdef long long b2_F_c      = (1LL << i2_F_c)

cdef long long b2_X_c      = (1LL << i2_X_c)
cdef long long b2_Y_c      = (1LL << i2_Y_c)

cdef long long b2_X2_c     = (1LL << i2_X2_c)
cdef long long b2_XY_c     = (1LL << i2_XY_c)
cdef long long b2_Y2_c     = (1LL << i2_Y2_c)

cdef long long b2_X3_c     = (1LL << i2_X3_c)
cdef long long b2_X2Y_c    = (1LL << i2_X2Y_c)
cdef long long b2_XY2_c    = (1LL << i2_XY2_c)
cdef long long b2_Y3_c     = (1LL << i2_Y3_c)

cdef long long b2_X4_c     = (1LL << i2_X4_c)
cdef long long b2_X3Y_c    = (1LL << i2_X3Y_c)
cdef long long b2_X2Y2_c   = (1LL << i2_X2Y2_c)
cdef long long b2_XY3_c    = (1LL << i2_XY3_c)
cdef long long b2_Y4_c     = (1LL << i2_Y4_c)

# 3D case
#
cdef long long b3_F_c      = (1LL << i3_F_c)

cdef long long b3_X_c      = (1LL << i3_X_c)
cdef long long b3_Y_c      = (1LL << i3_Y_c)
cdef long long b3_Z_c      = (1LL << i3_Z_c)

cdef long long b3_X2_c     = (1LL << i3_X2_c)
cdef long long b3_XY_c     = (1LL << i3_XY_c)
cdef long long b3_Y2_c     = (1LL << i3_Y2_c)
cdef long long b3_YZ_c     = (1LL << i3_YZ_c)
cdef long long b3_Z2_c     = (1LL << i3_Z2_c)
cdef long long b3_XZ_c     = (1LL << i3_XZ_c)

cdef long long b3_X3_c     = (1LL << i3_X3_c)
cdef long long b3_X2Y_c    = (1LL << i3_X2Y_c)
cdef long long b3_XY2_c    = (1LL << i3_XY2_c)
cdef long long b3_Y3_c     = (1LL << i3_Y3_c)
cdef long long b3_Y2Z_c    = (1LL << i3_Y2Z_c)
cdef long long b3_YZ2_c    = (1LL << i3_YZ2_c)
cdef long long b3_Z3_c     = (1LL << i3_Z3_c)
cdef long long b3_XZ2_c    = (1LL << i3_XZ2_c)
cdef long long b3_X2Z_c    = (1LL << i3_X2Z_c)
cdef long long b3_XYZ_c    = (1LL << i3_XYZ_c)

cdef long long b3_X4_c     = (1LL << i3_X4_c)
cdef long long b3_X3Y_c    = (1LL << i3_X3Y_c)
cdef long long b3_X2Y2_c   = (1LL << i3_X2Y2_c)
cdef long long b3_XY3_c    = (1LL << i3_XY3_c)
cdef long long b3_Y4_c     = (1LL << i3_Y4_c)
cdef long long b3_Y3Z_c    = (1LL << i3_Y3Z_c)
cdef long long b3_Y2Z2_c   = (1LL << i3_Y2Z2_c)
cdef long long b3_YZ3_c    = (1LL << i3_YZ3_c)
cdef long long b3_Z4_c     = (1LL << i3_Z4_c)
cdef long long b3_XZ3_c    = (1LL << i3_XZ3_c)
cdef long long b3_X2Z2_c   = (1LL << i3_X2Z2_c)
cdef long long b3_X3Z_c    = (1LL << i3_X3Z_c)
cdef long long b3_X2YZ_c   = (1LL << i3_X2YZ_c)
cdef long long b3_XY2Z_c   = (1LL << i3_XY2Z_c)
cdef long long b3_XYZ2_c   = (1LL << i3_XYZ2_c)


#################################################
# Python wrapper
#################################################

# Algorithms for the solve step (expert mode).
#
ALGO_BASIC     = ALGO_BASIC_c
ALGO_ITERATIVE = ALGO_ITERATIVE_c

# Weighting methods.
#
WEIGHT_UNIFORM = WEIGHT_UNIFORM_c
WEIGHT_CENTER  = WEIGHT_CENTER_c

# DOF index in the array f.
#
# These are ordered in increasing order of number of differentiations, so that if only first derivatives
# are required, the DOF array can be simply truncated after the first derivatives.
#
# To avoid gaps in the numbering, this requires separate DOF orderings for the 1D, 2D and 3D cases.
#
# (The other logical possibility would be function value first, then x-related, then y-related, then mixed,
#  but then the case of "first derivatives only" requires changes to the ordering to avoid gaps.
#  Specifying different orderings for different numbers of space dimensions is conceptually cleaner
#  of the two possibilities.)

# 1D case
#
i1_F  = i1_F_c
i1_X  = i1_X_c
i1_X2 = i1_X2_c
i1_X3 = i1_X3_c
i1_X4 = i1_X4_c

i1_1st_end = i1_1st_end_c
i1_2nd_end = i1_2nd_end_c
i1_3rd_end = i1_3rd_end_c
i1_4th_end = i1_4th_end_c

SIZE1 = SIZE1_c


# 2D case
#
i2_F     = i2_F_c

i2_X     = i2_X_c
i2_Y     = i2_Y_c

i2_X2    = i2_X2_c
i2_XY    = i2_XY_c
i2_Y2    = i2_Y2_c

i2_X3    = i2_X3_c
i2_X2Y   = i2_X2Y_c
i2_XY2   = i2_XY2_c
i2_Y3    = i2_Y3_c

i2_X4    = i2_X4_c
i2_X3Y   = i2_X3Y_c
i2_X2Y2  = i2_X2Y2_c
i2_XY3   = i2_XY3_c
i2_Y4    = i2_Y4_c

i2_1st_end = i2_1st_end_c
i2_2nd_end = i2_2nd_end_c
i2_3rd_end = i2_3rd_end_c
i2_4th_end = i2_4th_end_c

SIZE2 = SIZE2_c


# 3D case
#
i3_F      = i3_F_c

i3_X      = i3_X_c
i3_Y      = i3_Y_c
i3_Z      = i3_Z_c

i3_X2     = i3_X2_c
i3_XY     = i3_XY_c
i3_Y2     = i3_Y2_c
i3_YZ     = i3_YZ_c
i3_Z2     = i3_Z2_c
i3_XZ     = i3_XZ_c

i3_X3     = i3_X3_c
i3_X2Y    = i3_X2Y_c
i3_XY2    = i3_XY2_c
i3_Y3     = i3_Y3_c
i3_Y2Z    = i3_Y2Z_c
i3_YZ2    = i3_YZ2_c
i3_Z3     = i3_Z3_c
i3_XZ2    = i3_XZ2_c
i3_X2Z    = i3_X2Z_c
i3_XYZ    = i3_XYZ_c

i3_X4     = i3_X4_c
i3_X3Y    = i3_X3Y_c
i3_X2Y2   = i3_X2Y2_c
i3_XY3    = i3_XY3_c
i3_Y4     = i3_Y4_c
i3_Y3Z    = i3_Y3Z_c
i3_Y2Z2   = i3_Y2Z2_c
i3_YZ3    = i3_YZ3_c
i3_Z4     = i3_Z4_c
i3_XZ3    = i3_XZ3_c
i3_X2Z2   = i3_X2Z2_c
i3_X3Z    = i3_X3Z_c
i3_X2YZ   = i3_X2YZ_c
i3_XY2Z   = i3_XY2Z_c
i3_XYZ2   = i3_XYZ2_c

i3_0th_end = i3_0th_end_c
i3_1st_end = i3_1st_end_c
i3_2nd_end = i3_2nd_end_c
i3_3rd_end = i3_3rd_end_c
i3_4th_end = i3_4th_end_c

SIZE3 = SIZE3_c


# bitmask constants for knowns.
#
# Knowns are eliminated algebraically from the equation system; if any knowns are specified,
# the system to be solved (for a point x_i) will be smaller than the full 6x6.
#
# The sensible default thing to do is to consider the function value F known, with all the
# derivatives unknown.
#
# Note that here "known" means "known at point x_i" (the point at which we wish to compute the derivatives).
#
# Function values (F) are always assumed known at all *neighbor* points x_k, since they are used
# for determining the local least-squares quadratic polynomial fit to the data. This fit is then used
# as local a surrogate model for the unknown function f; in WLSQM, the derivatives are actually computed
# from the surrogate.
#
# The option to have the function value (F) as an unknown is useful with Neumann BCs, if the neighborhoods
# of the Neumann boundary points are chosen so that each Neumann boundary point only uses neighbors from
# the interior of the domain. This gives the possibility to leave F free at all Neumann boundary points,
# while prescribing only a derivative.
#
# (In practice, at slanted (i.e. not coordinate axis aligned) boundaries, local (tangent, normal)
#  coordinates must be used; i.e., the coordinate system in which the derivatives are to be computed
#  must be rotated to match the orientation of the boundary. This makes Y the normal derivative,
#  which can then be prescribed using this mechanism, while leaving the function value F free.)

# 1D case
#
b1_F  = b1_F_c
b1_X  = b1_X_c
b1_X2 = b1_X2_c
b1_X3 = b1_X3_c
b1_X4 = b1_X4_c


# 2D case
#
b2_F     = b2_F_c

b2_X     = b2_X_c
b2_Y     = b2_Y_c

b2_X2    = b2_X2_c
b2_XY    = b2_XY_c
b2_Y2    = b2_Y2_c

b2_X3    = b2_X3_c
b2_X2Y   = b2_X2Y_c
b2_XY2   = b2_XY2_c
b2_Y3    = b2_Y3_c

b2_X4    = b2_X4_c
b2_X3Y   = b2_X3Y_c
b2_X2Y2  = b2_X2Y2_c
b2_XY3   = b2_XY3_c
b2_Y4    = b2_Y4_c


# 3D case
#
b3_F      = b3_F_c

b3_X      = b3_X_c
b3_Y      = b3_Y_c
b3_Z      = b3_Z_c

b3_X2     = b3_X2_c
b3_XY     = b3_XY_c
b3_Y2     = b3_Y2_c
b3_YZ     = b3_YZ_c
b3_Z2     = b3_Z2_c
b3_XZ     = b3_XZ_c

b3_X3     = b3_X3_c
b3_X2Y    = b3_X2Y_c
b3_XY2    = b3_XY2_c
b3_Y3     = b3_Y3_c
b3_Y2Z    = b3_Y2Z_c
b3_YZ2    = b3_YZ2_c
b3_Z3     = b3_Z3_c
b3_XZ2    = b3_XZ2_c
b3_X2Z    = b3_X2Z_c
b3_XYZ    = b3_XYZ_c

b3_X4     = b3_X4_c
b3_X3Y    = b3_X3Y_c
b3_X2Y2   = b3_X2Y2_c
b3_XY3    = b3_XY3_c
b3_Y4     = b3_Y4_c
b3_Y3Z    = b3_Y3Z_c
b3_Y2Z2   = b3_Y2Z2_c
b3_YZ3    = b3_YZ3_c
b3_Z4     = b3_Z4_c
b3_XZ3    = b3_XZ3_c
b3_X2Z2   = b3_X2Z2_c
b3_X3Z    = b3_X3Z_c
b3_X2YZ   = b3_X2YZ_c
b3_XY2Z   = b3_XY2Z_c
b3_XYZ2   = b3_XYZ2_c

