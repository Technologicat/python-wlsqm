# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# C-level definitions for Cython.
#
# This file contains only the declarations; the actual values are set in the .pyx source for wlsqm.fitter.defs.
#
# The suffix of _c means "visible at the C level in Cython"; it is used to distinguish
# the typed C constants from the Python-accessible objects also defined in the .pyx source.
#
# JJ 2016-11-30

# Algorithms for the solve step (expert mode).
#
cdef int ALGO_BASIC_c      # fit just once
cdef int ALGO_ITERATIVE_c  # fit with iterative refinement to mitigate roundoff

# Weighting methods.
#
cdef int WEIGHT_UNIFORM_c
cdef int WEIGHT_CENTER_c

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
cdef int i1_F_c
cdef int i1_X_c
cdef int i1_X2_c
cdef int i1_X3_c
cdef int i1_X4_c

cdef int i1_0th_end_c  # one-past end of zeroth-order case
cdef int i1_1st_end_c  # one-past end of first-order case
cdef int i1_2nd_end_c  # one-past-end of second-order case
cdef int i1_3rd_end_c  # one-past-end of third-order case
cdef int i1_4th_end_c  # one-past-end of fourth-order case

cdef int SIZE1_c  # maximum possible number of DOFs, 1D case

# 2D case
#
cdef int i2_F_c

cdef int i2_X_c
cdef int i2_Y_c

cdef int i2_X2_c
cdef int i2_XY_c
cdef int i2_Y2_c

cdef int i2_X3_c
cdef int i2_X2Y_c
cdef int i2_XY2_c
cdef int i2_Y3_c

cdef int i2_X4_c
cdef int i2_X3Y_c
cdef int i2_X2Y2_c
cdef int i2_XY3_c
cdef int i2_Y4_c

cdef int i2_0th_end_c  # one-past end of zeroth-order case
cdef int i2_1st_end_c  # one-past end of first-order case
cdef int i2_2nd_end_c  # one-past-end of second-order case
cdef int i2_3rd_end_c  # one-past-end of third-order case
cdef int i2_4th_end_c  # one-past-end of fourth-order case

cdef int SIZE2_c  # maximum possible number of DOFs, 2D case

# 3D case
#
cdef int i3_F_c

cdef int i3_X_c
cdef int i3_Y_c
cdef int i3_Z_c

cdef int i3_X2_c
cdef int i3_XY_c
cdef int i3_Y2_c
cdef int i3_YZ_c
cdef int i3_Z2_c
cdef int i3_XZ_c

cdef int i3_X3_c
cdef int i3_X2Y_c
cdef int i3_XY2_c
cdef int i3_Y3_c
cdef int i3_Y2Z_c
cdef int i3_YZ2_c
cdef int i3_Z3_c
cdef int i3_XZ2_c
cdef int i3_X2Z_c
cdef int i3_XYZ_c

cdef int i3_X4_c
cdef int i3_X3Y_c
cdef int i3_X2Y2_c
cdef int i3_XY3_c
cdef int i3_Y4_c
cdef int i3_Y3Z_c
cdef int i3_Y2Z2_c
cdef int i3_YZ3_c
cdef int i3_Z4_c
cdef int i3_XZ3_c
cdef int i3_X2Z2_c
cdef int i3_X3Z_c
cdef int i3_X2YZ_c
cdef int i3_XY2Z_c
cdef int i3_XYZ2_c

cdef int i3_0th_end_c  # one-past end of zeroth-order case
cdef int i3_1st_end_c  # one-past end of first-order case
cdef int i3_2nd_end_c  # one-past-end of second-order case
cdef int i3_3rd_end_c  # one-past-end of third-order case
cdef int i3_4th_end_c  # one-past-end of fourth-order case

cdef int SIZE3_c  # maximum possible number of DOFs, 3D case


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
cdef long long b1_F_c
cdef long long b1_X_c
cdef long long b1_X2_c
cdef long long b1_X3_c
cdef long long b1_X4_c

# 2D case
#
cdef long long b2_F_c

cdef long long b2_X_c
cdef long long b2_Y_c

cdef long long b2_X2_c
cdef long long b2_XY_c
cdef long long b2_Y2_c

cdef long long b2_X3_c
cdef long long b2_X2Y_c
cdef long long b2_XY2_c
cdef long long b2_Y3_c

cdef long long b2_X4_c
cdef long long b2_X3Y_c
cdef long long b2_X2Y2_c
cdef long long b2_XY3_c
cdef long long b2_Y4_c

# 3D case
#
cdef long long b3_F_c

cdef long long b3_X_c
cdef long long b3_Y_c
cdef long long b3_Z_c

cdef long long b3_X2_c
cdef long long b3_XY_c
cdef long long b3_Y2_c
cdef long long b3_YZ_c
cdef long long b3_Z2_c
cdef long long b3_XZ_c

cdef long long b3_X3_c
cdef long long b3_X2Y_c
cdef long long b3_XY2_c
cdef long long b3_Y3_c
cdef long long b3_Y2Z_c
cdef long long b3_YZ2_c
cdef long long b3_Z3_c
cdef long long b3_XZ2_c
cdef long long b3_X2Z_c
cdef long long b3_XYZ_c

cdef long long b3_X4_c
cdef long long b3_X3Y_c
cdef long long b3_X2Y2_c
cdef long long b3_XY3_c
cdef long long b3_Y4_c
cdef long long b3_Y3Z_c
cdef long long b3_Y2Z2_c
cdef long long b3_YZ3_c
cdef long long b3_Z4_c
cdef long long b3_XZ3_c
cdef long long b3_X2Z2_c
cdef long long b3_X3Z_c
cdef long long b3_X2YZ_c
cdef long long b3_XY2Z_c
cdef long long b3_XYZ2_c

