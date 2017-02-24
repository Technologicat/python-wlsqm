# -*- coding: utf-8 -*-
#
# WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.
#
# Evaluation of Taylor expansions and general polynomials up to 4th order in 1D, 2D and 3D.
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

from __future__ import division
from __future__ import absolute_import

# BUG in Cython 0.20.1post0: /usr/lib/python2.7/dist-packages/Cython/Includes/libc/math.pxd
# defines fma(double x, double y), but it should be fma(double x, double y, double z)
# so we import it manually.
#
#from libc.math cimport fma   # this works in newer Cythons, the bug has been fixed
cdef extern from "<math.h>" nogil:
    double fma(double x, double y, double z)

cimport wlsqm.wlsqm2.wlsqm2_defs as defs # C constants

####################################################
# Polynomial evaluation
####################################################

# These are implemented as a separate layer in order to allow us to re-use the same implementation
# for interpolating both the function value and the derivatives of the model.


# Evaluate an up to 4th order Taylor series expansion in 3D space, with its origin at (xi,yi,zi).
#
# This version uses "partially baked" coefficients, already accounting for the constant factors in the Taylor series!
# (This allows expressing the derivative information at the point xi in the raw coefficient data.)
#
# The implementation uses a symmetric Horner-like form with fused multiply-adds.
# (Note however that some of the symmetry is lost due to the way the partial results are summed.)
#
# order : in, the order of the expansion (0,1,2,3 or 4)
# fi    : in, coefficient array ("order" determines the number of entries, no separate size parameter needed)
#         The ordering of the coefficients follows the numbering of the i3_* constants in wlsqm2_defs.pyx.
#         Here fi[ i3_F_c ] is f at xi, fi[ i3_X ] is df/fx at xi, fi[ i3_Y ] is df/dy at xi, fi[ i3_X2 ] is d2f/dx2 at xi, ...
# xi    : in, origin of the model, x component
# yi    : in, origin of the model, y component
# zi    : in, origin of the model, z component
# x     : in, the points where to evaluate the Taylor expansion
# out   : out, array of size (x.shape[0],); the result
#
# Return value: 0 on success, anything else indicates failure
#
cdef int evaluate_taylor_expansion_3D( int order, double* fi, double xi, double yi, double zi, double[::view.generic,::view.contiguous] x, double* out ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx, dy, dz
    cdef double dxdy, dydz, dxdz
    cdef double acc1, acc2  # accumulators
    cdef double resX, resY, resZ, resXY, resYZ, resXZ, resXYZ  # intermediate results

    if order == 4:
        for k in range(n):
            dx   = x[k,0] - xi
            dy   = x[k,1] - yi
            dz   = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dz2  = dz*dz
#            dx3  = dx2*dx
#            dy3  = dy2*dy
#            dz3  = dz2*dz
#            dxdy = dx*dy
#            dydz = dy*dz
#            dxdz = dx*dz
#            out[k] = fi[ i3_F_c ] + dx*fi[ i3_X_c ] + dy*fi[ i3_Y_c ] + dz*fi[ i3_Z_c ] \
#
#                   + 0.5*dx2*fi[ i3_X2_c ] + dxdy*fi[ i3_XY_c ] + 0.5*dy2*fi[ i3_Y2_c ] + dydz*fi[ i3_YZ_c ] + 0.5*dz2*fi[ i3_Z2_c ] + dxdz*fi[ i3_XZ_c ] \
#
#                   + onesixth*dx3*fi[ i3_X3_c ] + 0.5*dx2*dy*fi[ i3_X2Y_c ] + 0.5*dx*dy2*fi[ i3_XY2_c ] + onesixth*dy3*fi[ i3_Y3_c ] \
#                   + 0.5*dy2*dz*fi[ i3_Y2Z_c ] + 0.5*dy*dz2*fi[ i3_YZ2_c ] + onesixth*dz3*fi[ i3_Z3_c ] + 0.5*dx*dz2*fi[ i3_XZ2_c ] + 0.5*dx2*dz*fi[ i3_X2Z_c ] + dx*dy*dz*fi[ i3_XYZ_c ] \
#
#                   + one24th*dx2*dx2*fi[ i3_X4_c ] + onesixth*dx3*dy*fi[ i3_X3Y_c ] + 0.25*dxdy*dxdy*fi[ i3_X2Y2_c ] + onesixth*dx*dy3*fi[ i3_XY3_c ] + one24th*dy2*dy2*fi[ i3_Y4_c ] \
#                   + onesixth*dy3*dz*fi[ i3_Y3Z_c ] + 0.25*dydz*dydz*fi[ i3_Y2Z2_c ] + onesixth*dy*dz3*fi[ i3_YZ3_c ] + one24th*dz2*dz2*fi[ i3_Z4_c ] + onesixth*dx*dz3*fi[ i3_XZ3_c ] \
#                   + 0.25*dxdz*dxdz*fi[ i3_X2Z2_c ] + onesixth*dx3*dz*fi[ i3_X3Z_c ] + 0.5*dx2*dydz*fi[ i3_X2YZ_c ] + 0.5*dy2*dxdz*fi[ i3_XY2Z_c ] + 0.5*dxdy*dz2*fi[ i3_XYZ2_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i3_F_c ] + dx*( fi[ i3_X_c ] + dx*( 0.5*( fi[ i3_X2_c ] + dy*fi[ i3_X2Y_c ] + dz*fi[ i3_X2Z_c ] + dydz*fi[ i3_X2YZ_c ] )
#                                                             + dx*( onesixth*( fi[ i3_X3_c ] + dy*fi[ i3_X3Y_c ] + dz*fi[ i3_X3Z_c ] )
#                                                                    + dx*( one24th*fi[ i3_X4_c ] ) ) ) ) \
#
#                                  + dy*( fi[ i3_Y_c ] + dy*( 0.5*( fi[ i3_Y2_c ] + dx*fi[ i3_XY2_c ] + dz*fi[ i3_Y2Z_c ] + dxdz*fi[ i3_XY2Z_c ] )
#                                                             + dy*( onesixth*( fi[ i3_Y3_c ] + dx*fi[ i3_XY3_c ] + dz*fi[ i3_Y3Z_c ] )
#                                                                    + dy*( one24th*fi[ i3_Y4_c ] ) ) ) ) \
#
#                                  + dz*( fi[ i3_Z_c ] + dz*( 0.5*( fi[ i3_Z2_c ] + dx*fi[ i3_XZ2_c ] + dy*fi[ i3_YZ2_c ] + dxdy*fi[ i3_XYZ2_c ] )
#                                                             + dz*( onesixth*( fi[ i3_Z3_c ] + dx*fi[ i3_XZ3_c ] + dy*fi[ i3_YZ3_c ] )
#                                                                    + dz*( one24th*fi[ i3_Z4_c ] ) ) ) ) \
#
#                                  + dxdy*( fi[ i3_XY_c ] + dxdy*0.25*fi[ i3_X2Y2_c ] ) \
#                                  + dydz*( fi[ i3_YZ_c ] + dydz*0.25*fi[ i3_Y2Z2_c ] ) \
#                                  + dxdz*( fi[ i3_XZ_c ] + dxdz*0.25*fi[ i3_X2Z2_c ] ) \
#
#                                  + dx*dy*dz*fi[ i3_XYZ_c ]

            # symmetric Horner-like form using fused multiply-add (fma):

            # "X" terms (the "..." in  dx*...  above)
            acc1   = fma( dy, fi[ defs.i3_X3Y_c ], fi[ defs.i3_X3_c ] )
            acc1   = fma( dz, fi[ defs.i3_X3Z_c ], acc1 )
            acc1  *= onesixth
            acc1   = fma( dx, one24th*fi[ defs.i3_X4_c ], acc1 )

            acc2   = fma( dy,   fi[ defs.i3_X2Y_c  ], fi[ defs.i3_X2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_X2Z_c  ], acc2 )
            acc2   = fma( dydz, fi[ defs.i3_X2YZ_c ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dx, acc1, acc2 )

            resX   = fma( dx, acc2, fi[ defs.i3_X_c ] )

            # "Y" terms
            acc1   = fma( dx, fi[ defs.i3_XY3_c ], fi[ defs.i3_Y3_c ] )
            acc1   = fma( dz, fi[ defs.i3_Y3Z_c ], acc1 )
            acc1  *= onesixth
            acc1   = fma( dy, one24th*fi[ defs.i3_Y4_c ], acc1 )

            acc2   = fma( dx,   fi[ defs.i3_XY2_c  ], fi[ defs.i3_Y2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_Y2Z_c  ], acc2 )
            acc2   = fma( dxdz, fi[ defs.i3_XY2Z_c ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dy, acc1, acc2 )

            resY   = fma( dy, acc2, fi[ defs.i3_Y_c ] )

            # "Z" terms
            acc1   = fma( dx, fi[ defs.i3_XZ3_c ], fi[ defs.i3_Z3_c ] )
            acc1   = fma( dy, fi[ defs.i3_YZ3_c ], acc1 )
            acc1  *= onesixth
            acc1   = fma( dz, one24th*fi[ defs.i3_Z4_c ], acc1 )

            acc2   = fma( dx,   fi[ defs.i3_XZ2_c  ], fi[ defs.i3_Z2_c ] )
            acc2   = fma( dy,   fi[ defs.i3_YZ2_c  ], acc2 )
            acc2   = fma( dxdy, fi[ defs.i3_XYZ2_c ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dz, acc1, acc2 )

            resZ   = fma( dz, acc2, fi[ defs.i3_Z_c ] )

            # "XY", "YZ", "XZ" terms
            resXY  = fma( dxdy, 0.25*fi[ defs.i3_X2Y2_c ], fi[ defs.i3_XY_c ] )
            resYZ  = fma( dydz, 0.25*fi[ defs.i3_Y2Z2_c ], fi[ defs.i3_YZ_c ] )
            resXZ  = fma( dxdz, 0.25*fi[ defs.i3_X2Z2_c ], fi[ defs.i3_XZ_c ] )

            # sum the contributions (in an approximately increasing order of magnitude)
            #
            acc1   = dx*dy*dz*fi[ defs.i3_XYZ_c ]
            acc1   = fma( dxdy,     resXY,               acc1 )
            acc1   = fma( dydz,     resYZ,               acc1 )
            acc1   = fma( dxdz,     resXZ,               acc1 )
            acc1   = fma( dx,       resX,                acc1 )
            acc1   = fma( dy,       resY,                acc1 )
            acc1   = fma( dz,       resZ,                acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 3:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dz2  = dz*dz
#            dx3  = dx2*dx
#            dy3  = dy2*dy
#            dz3  = dz2*dz
#            dxdy = dx*dy
#            dydz = dy*dz
#            dxdz = dx*dz
#            out[k] = fi[ i3_F_c ] + dx*fi[ i3_X_c ] + dy*fi[ i3_Y_c ] + dz*fi[ i3_Z_c ] \
#
#                   + 0.5*dx2*fi[ i3_X2_c ] + dxdy*fi[ i3_XY_c ] + 0.5*dy2*fi[ i3_Y2_c ] + dydz*fi[ i3_YZ_c ] + 0.5*dz2*fi[ i3_Z2_c ] + dxdz*fi[ i3_XZ_c ] \
#
#                   + onesixth*dx3*fi[ i3_X3_c ] + 0.5*dx2*dy*fi[ i3_X2Y_c ] + 0.5*dx*dy2*fi[ i3_XY2_c ] + onesixth*dy3*fi[ i3_Y3_c ] \
#                   + 0.5*dy2*dz*fi[ i3_Y2Z_c ] + 0.5*dy*dz2*fi[ i3_YZ2_c ] + onesixth*dz3*fi[ i3_Z3_c ] + 0.5*dx*dz2*fi[ i3_XZ2_c ] + 0.5*dx2*dz*fi[ i3_X2Z_c ] + dx*dy*dz*fi[ i3_XYZ_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i3_F_c ] + dx*( fi[ i3_X_c ] + dx*( 0.5*( fi[ i3_X2_c ] + dy*fi[ i3_X2Y_c ] + dz*fi[ i3_X2Z_c ] )
#                                                             + dx*( onesixth*( fi[ i3_X3_c ] ) ) ) ) \
#
#                                  + dy*( fi[ i3_Y_c ] + dy*( 0.5*( fi[ i3_Y2_c ] + dx*fi[ i3_XY2_c ] + dz*fi[ i3_Y2Z_c ] )
#                                                             + dy*( onesixth*( fi[ i3_Y3_c ] ) ) ) ) \
#
#                                  + dz*( fi[ i3_Z_c ] + dz*( 0.5*( fi[ i3_Z2_c ] + dx*fi[ i3_XZ2_c ] + dy*fi[ i3_YZ2_c ] )
#                                                             + dz*( onesixth*( fi[ i3_Z3_c ] ) ) ) ) \
#
#                                  + dxdy*( fi[ i3_XY_c ] ) \
#                                  + dydz*( fi[ i3_YZ_c ] ) \
#                                  + dxdz*( fi[ i3_XZ_c ] ) \
#
#                                  + dx*dy*dz*fi[ i3_XYZ_c ]

            # "X" terms (the "..." in  dx*...  above)
            acc2   = fma( dy,   fi[ defs.i3_X2Y_c  ], fi[ defs.i3_X2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_X2Z_c  ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dx, onesixth*fi[ defs.i3_X3_c ], acc2 )

            resX   = fma( dx, acc2, fi[ defs.i3_X_c ] )

            # "Y" terms
            acc2   = fma( dx,   fi[ defs.i3_XY2_c  ], fi[ defs.i3_Y2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_Y2Z_c  ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dy, onesixth*fi[ defs.i3_Y3_c ], acc2 )

            resY   = fma( dy, acc2, fi[ defs.i3_Y_c ] )

            # "Z" terms
            acc2   = fma( dx,   fi[ defs.i3_XZ2_c  ], fi[ defs.i3_Z2_c ] )
            acc2   = fma( dy,   fi[ defs.i3_YZ2_c  ], acc2 )
            acc2  *= 0.5
            acc2   = fma( dz, onesixth*fi[ defs.i3_Z3_c ], acc2 )

            resZ   = fma( dz, acc2, fi[ defs.i3_Z_c ] )

            # sum the contributions
            #
            acc1   = dx*dy*dz*fi[ defs.i3_XYZ_c ]
            acc1   = fma( dxdy,     fi[ defs.i3_XY_c  ], acc1 )
            acc1   = fma( dydz,     fi[ defs.i3_YZ_c  ], acc1 )
            acc1   = fma( dxdz,     fi[ defs.i3_XZ_c  ], acc1 )
            acc1   = fma( dx,       resX,                acc1 )
            acc1   = fma( dy,       resY,                acc1 )
            acc1   = fma( dz,       resZ,                acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 2:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dz2  = dz*dz
#            dxdy = dx*dy
#            dydz = dy*dz
#            dxdz = dx*dz
#            out[k] = fi[ i3_F_c ] + dx*fi[ i3_X_c ] + dy*fi[ i3_Y_c ] + dz*fi[ i3_Z_c ] \
#                   + 0.5*dx2*fi[ i3_X2_c ] + dxdy*fi[ i3_XY_c ] + 0.5*dy2*fi[ i3_Y2_c ] + dydz*fi[ i3_YZ_c ] + 0.5*dz2*fi[ i3_Z2_c ] + dxdz*fi[ i3_XZ_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i3_F_c ] + dx*( fi[ i3_X_c ] + dx*( 0.5*( fi[ i3_X2_c ] ) ) ) \
#                                  + dy*( fi[ i3_Y_c ] + dy*( 0.5*( fi[ i3_Y2_c ] ) ) ) \
#                                  + dz*( fi[ i3_Z_c ] + dz*( 0.5*( fi[ i3_Z2_c ] ) ) ) \
#
#                                  + dxdy*( fi[ i3_XY_c ] ) \
#                                  + dydz*( fi[ i3_YZ_c ] ) \
#                                  + dxdz*( fi[ i3_XZ_c ] )

            resX   = fma( dx, 0.5*fi[ defs.i3_X2_c ], fi[ defs.i3_X_c ] )
            resY   = fma( dy, 0.5*fi[ defs.i3_Y2_c ], fi[ defs.i3_Y_c ] )
            resZ   = fma( dz, 0.5*fi[ defs.i3_Z2_c ], fi[ defs.i3_Z_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i3_XY_c ]
            acc1   = fma( dydz, fi[ defs.i3_YZ_c ], acc1 )
            acc1   = fma( dxdz, fi[ defs.i3_XZ_c ], acc1 )
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1   = fma( dz,   resZ,               acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 1:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi
#            # for documentation only:
#            out[k] = fi[ i3_F_c ] + dx*fi[ i3_X_c ] + dy*fi[ i3_Y_c ] + dz*fi[ i3_Z_c ]

            # sum the contributions
            #
            acc1   = dx*fi[ defs.i3_X_c ]
            acc1   = fma( dy, fi[ defs.i3_Y_c ], acc1 )
            acc1   = fma( dz, fi[ defs.i3_Z_c ], acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i3_F_c ]

    return 0


# Same strategy as above, but for a generic polynomial (no partially baked coefficients).
#
# Now fi[ i3_F_c ] is the constant term, fi[ i3_X ] is the coefficient of (x - xi), fi[ i3_Y ] is the coefficient of (y - yi), fi[ i3_X2 ] is the coefficient of (x - xi)**2, ...
#
cdef int evaluate_polynomial_3D( int order, double* fi, double xi, double yi, double zi, double[::view.generic,::view.contiguous] x, double* out ) nogil:
    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx, dy, dz
    cdef double dxdy, dydz, dxdz
    cdef double acc1, acc2  # accumulators
    cdef double resX, resY, resZ, resXY, resYZ, resXZ, resXYZ  # intermediate results

    if order == 4:
        for k in range(n):
            dx   = x[k,0] - xi
            dy   = x[k,1] - yi
            dz   = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz

            # symmetric Horner-like form using fused multiply-add (fma):

            # "X" terms
            acc1   = fma( dy,   fi[ defs.i3_X3Y_c  ], fi[ defs.i3_X3_c ] )
            acc1   = fma( dz,   fi[ defs.i3_X3Z_c  ], acc1 )
            acc1   = fma( dx,   fi[ defs.i3_X4_c   ], acc1 )

            acc2   = fma( dy,   fi[ defs.i3_X2Y_c  ], fi[ defs.i3_X2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_X2Z_c  ], acc2 )
            acc2   = fma( dydz, fi[ defs.i3_X2YZ_c ], acc2 )
            acc2   = fma( dx,   acc1,                 acc2 )

            resX   = fma( dx, acc2, fi[ defs.i3_X_c ] )

            # "Y" terms
            acc1   = fma( dx,   fi[ defs.i3_XY3_c  ], fi[ defs.i3_Y3_c ] )
            acc1   = fma( dz,   fi[ defs.i3_Y3Z_c  ], acc1 )
            acc1   = fma( dy,   fi[ defs.i3_Y4_c   ], acc1 )

            acc2   = fma( dx,   fi[ defs.i3_XY2_c  ], fi[ defs.i3_Y2_c ] )
            acc2   = fma( dz,   fi[ defs.i3_Y2Z_c  ], acc2 )
            acc2   = fma( dxdz, fi[ defs.i3_XY2Z_c ], acc2 )
            acc2   = fma( dy,   acc1,                 acc2 )

            resY   = fma( dy, acc2, fi[ defs.i3_Y_c ] )

            # "Z" terms
            acc1   = fma( dx,   fi[ defs.i3_XZ3_c  ], fi[ defs.i3_Z3_c ] )
            acc1   = fma( dy,   fi[ defs.i3_YZ3_c  ], acc1 )
            acc1   = fma( dz,   fi[ defs.i3_Z4_c   ], acc1 )

            acc2   = fma( dx,   fi[ defs.i3_XZ2_c  ], fi[ defs.i3_Z2_c ] )
            acc2   = fma( dy,   fi[ defs.i3_YZ2_c  ], acc2 )
            acc2   = fma( dxdy, fi[ defs.i3_XYZ2_c ], acc2 )
            acc2   = fma( dz,   acc1,                 acc2 )

            resZ   = fma( dz, acc2, fi[ defs.i3_Z_c ] )

            # "XY", "YZ", "XZ" terms
            resXY  = fma( dxdy, fi[ defs.i3_X2Y2_c ], fi[ defs.i3_XY_c ] )
            resYZ  = fma( dydz, fi[ defs.i3_Y2Z2_c ], fi[ defs.i3_YZ_c ] )
            resXZ  = fma( dxdz, fi[ defs.i3_X2Z2_c ], fi[ defs.i3_XZ_c ] )

            # sum the contributions (in an approximately increasing order of magnitude)
            #
            acc1   = dx*dy*dz*fi[ defs.i3_XYZ_c ]
            acc1   = fma( dxdy,     resXY,               acc1 )
            acc1   = fma( dydz,     resYZ,               acc1 )
            acc1   = fma( dxdz,     resXZ,               acc1 )
            acc1   = fma( dx,       resX,                acc1 )
            acc1   = fma( dy,       resY,                acc1 )
            acc1   = fma( dz,       resZ,                acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 3:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz

            # "X" terms
            acc2   = fma( dy, fi[ defs.i3_X2Y_c ], fi[ defs.i3_X2_c ] )
            acc2   = fma( dz, fi[ defs.i3_X2Z_c ], acc2 )
            acc2   = fma( dx, fi[ defs.i3_X3_c  ], acc2 )

            resX   = fma( dx, acc2, fi[ defs.i3_X_c ] )

            # "Y" terms
            acc2   = fma( dx, fi[ defs.i3_XY2_c ], fi[ defs.i3_Y2_c ] )
            acc2   = fma( dz, fi[ defs.i3_Y2Z_c ], acc2 )
            acc2   = fma( dy, fi[ defs.i3_Y3_c  ], acc2 )

            resY   = fma( dy, acc2, fi[ defs.i3_Y_c ] )

            # "Z" terms
            acc2   = fma( dx, fi[ defs.i3_XZ2_c ], fi[ defs.i3_Z2_c ] )
            acc2   = fma( dy, fi[ defs.i3_YZ2_c ], acc2 )
            acc2   = fma( dz, fi[ defs.i3_Z3_c  ], acc2 )

            resZ   = fma( dz, acc2, fi[ defs.i3_Z_c ] )

            # sum the contributions
            #
            acc1   = dx*dy*dz*fi[ defs.i3_XYZ_c ]
            acc1   = fma( dxdy,     fi[ defs.i3_XY_c  ], acc1 )
            acc1   = fma( dydz,     fi[ defs.i3_YZ_c  ], acc1 )
            acc1   = fma( dxdz,     fi[ defs.i3_XZ_c  ], acc1 )
            acc1   = fma( dx,       resX,                acc1 )
            acc1   = fma( dy,       resY,                acc1 )
            acc1   = fma( dz,       resZ,                acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 2:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi
            dxdy = dx*dy
            dydz = dy*dz
            dxdz = dx*dz

            resX   = fma( dx, fi[ defs.i3_X2_c ], fi[ defs.i3_X_c ] )
            resY   = fma( dy, fi[ defs.i3_Y2_c ], fi[ defs.i3_Y_c ] )
            resZ   = fma( dz, fi[ defs.i3_Z2_c ], fi[ defs.i3_Z_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i3_XY_c ]
            acc1   = fma( dydz, fi[ defs.i3_YZ_c ], acc1 )
            acc1   = fma( dxdz, fi[ defs.i3_XZ_c ], acc1 )
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1   = fma( dz,   resZ,               acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    elif order == 1:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dz = x[k,2] - zi

            # sum the contributions
            #
            acc1   = dx*fi[ defs.i3_X_c ]
            acc1   = fma( dy, fi[ defs.i3_Y_c ], acc1 )
            acc1   = fma( dz, fi[ defs.i3_Z_c ], acc1 )
            acc1  += fi[ defs.i3_F_c ]

            out[k] = acc1

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i3_F_c ]

    return 0


# Evaluate an up to 4th order Taylor series expansion in the plane, with its origin at (xi,yi).
#
# This version uses "partially baked" coefficients, already accounting for the constant factors in the Taylor series!
# (This allows expressing the derivative information at the point xi in the raw coefficient data.)
#
# The implementation uses a symmetric Horner-like form with fused multiply-adds.
# (Note however that some of the symmetry is lost due to the way the partial results are summed.)
#
# order : in, the order of the expansion (0,1,2,3 or 4)
# fi    : in, coefficient array ("order" determines the number of entries, no separate size parameter needed)
#         The ordering of the coefficients follows the numbering of the i2_* constants in wlsqm2_defs.pyx.
#         Here fi[ i2_F_c ] is f at xi, fi[ i2_X ] is df/fx at xi, fi[ i2_Y ] is df/dy at xi, fi[ i2_X2 ] is d2f/dx2 at xi, ...
# xi    : in, origin of the model, x component
# yi    : in, origin of the model, y component
# x     : in, the points where to evaluate the Taylor expansion
# out   : out, array of size (x.shape[0],); the result
#
# Return value: 0 on success, anything else indicates failure
#
cdef int evaluate_taylor_expansion_2D( int order, double* fi, double xi, double yi, double[::view.generic,::view.contiguous] x, double* out ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx, dy
    cdef double dxdy
    cdef double acc1, acc2  # accumulators
    cdef double resX, resY, resXY  # intermediate results

    if order == 4:
        for k in range(n):
            dx   = x[k,0] - xi
            dy   = x[k,1] - yi
            dxdy = dx*dy
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dx3  = dx2*dx
#            dy3  = dy2*dy
#            dxdy = dx*dy
#            out[k] = fi[ i2_F_c ] + dx*fi[ i2_X_c ] + dy*fi[ i2_Y_c ] \
#
#                   + 0.5*dx2*fi[ i2_X2_c ] + dxdy*fi[ i2_XY_c ] + 0.5*dy2*fi[ i2_Y2_c ] \
#
#                   + onesixth*dx3*fi[ i2_X3_c ] + 0.5*dx2*dy*fi[ i2_X2Y_c ] + 0.5*dx*dy2*fi[ i2_XY2_c ] + onesixth*dy3*fi[ i2_Y3_c ] \
#
#                   + one24th*dx2*dx2*fi[ i2_X4_c ] + onesixth*dx3*dy*fi[ i2_X3Y_c ] + 0.25*dxdy*dxdy*fi[ i2_X2Y2_c ] + onesixth*dx*dy3*fi[ i2_XY3_c ] + one24th*dy2*dy2*fi[ i2_Y4_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i2_F_c ] + dx*( fi[ i2_X_c ] + dx*( 0.5*( fi[ i2_X2_c ] + dy*fi[ i2_X2Y_c ] )
#                                                             + dx*( onesixth*( fi[ i2_X3_c ] + dy*fi[ i2_X3Y_c ] )
#                                                                    + dx*( one24th*fi[ i2_X4_c ] ) ) ) ) \
#
#                                  + dy*( fi[ i2_Y_c ] + dy*( 0.5*( fi[ i2_Y2_c ] + dx*fi[ i2_XY2_c ] )
#                                                             + dy*( onesixth*( fi[ i2_Y3_c ] + dx*fi[ i2_XY3_c ] )
#                                                                    + dy*( one24th*fi[ i2_Y4_c ] ) ) ) ) \
#
#                                  + dxdy*( fi[ i2_XY_c ] + dxdy*0.25*fi[ i2_X2Y2_c ] )

            # symmetric Horner-like form using fused multiply-add (fma):

            # "X" terms (the "..." in  dx*...  above)
            acc1   = fma( dy, fi[ defs.i2_X3Y_c ], fi[ defs.i2_X3_c ] )
            acc1  *= onesixth
            acc1   = fma( dx, one24th*fi[ defs.i2_X4_c ], acc1 )

            acc2   = fma( dy,   fi[ defs.i2_X2Y_c  ], fi[ defs.i2_X2_c ] )
            acc2  *= 0.5
            acc2   = fma( dx, acc1, acc2 )

            resX   = fma( dx, acc2, fi[ defs.i2_X_c ] )

            # "Y" terms
            acc1   = fma( dx, fi[ defs.i2_XY3_c ], fi[ defs.i2_Y3_c ] )
            acc1  *= onesixth
            acc1   = fma( dy, one24th*fi[ defs.i2_Y4_c ], acc1 )

            acc2   = fma( dx,   fi[ defs.i2_XY2_c  ], fi[ defs.i2_Y2_c ] )
            acc2  *= 0.5
            acc2   = fma( dy, acc1, acc2 )

            resY   = fma( dy, acc2, fi[ defs.i2_Y_c ] )

            # "XY" terms
            resXY  = fma( dxdy, 0.25*fi[ defs.i2_X2Y2_c ], fi[ defs.i2_XY_c ] )

            # sum the contributions (in an approximately increasing order of magnitude)
            #
            acc1   = dxdy*resXY
            acc1   = fma( dx,   resX,  acc1 )
            acc1   = fma( dy,   resY,  acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 3:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dxdy = dx*dy
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dx3  = dx2*dx
#            dy3  = dy2*dy
#            dxdy = dx*dy
#            out[k] = fi[ i2_F_c ] + dx*fi[ i2_X_c ] + dy*fi[ i2_Y_c ] \
#
#                   + 0.5*dx2*fi[ i2_X2_c ] + dxdy*fi[ i2_XY_c ] + 0.5*dy2*fi[ i2_Y2_c ] \
#
#                   + onesixth*dx3*fi[ i2_X3_c ] + 0.5*dx2*dy*fi[ i2_X2Y_c ] + 0.5*dx*dy2*fi[ i2_XY2_c ] + onesixth*dy3*fi[ i2_Y3_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i2_F_c ] + dx*( fi[ i2_X_c ] + dx*( 0.5*( fi[ i2_X2_c ] + dy*fi[ i2_X2Y_c ] )
#                                                             + dx*( onesixth*( fi[ i2_X3_c ] ) ) ) ) \
#
#                                  + dy*( fi[ i2_Y_c ] + dy*( 0.5*( fi[ i2_Y2_c ] + dx*fi[ i2_XY2_c ] )
#                                                             + dy*( onesixth*( fi[ i2_Y3_c ] ) ) ) ) \
#
#                                  + dxdy*( fi[ i2_XY_c ] )

            # "X" terms (the "..." in  dx*...  above)
            acc2   = fma( dy,   fi[ defs.i2_X2Y_c  ], fi[ defs.i2_X2_c ] )
            acc2  *= 0.5
            acc2   = fma( dx, onesixth*fi[ defs.i2_X3_c ], acc2 )

            resX   = fma( dx, acc2, fi[ defs.i2_X_c ] )

            # "Y" terms
            acc2   = fma( dx,   fi[ defs.i2_XY2_c  ], fi[ defs.i2_Y2_c ] )
            acc2  *= 0.5
            acc2   = fma( dy, onesixth*fi[ defs.i2_Y3_c ], acc2 )

            resY   = fma( dy, acc2, fi[ defs.i2_Y_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i2_XY_c ]
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 2:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dxdy = dx*dy
#            # for documentation only:
#
#            # naive form:
#            dx2  = dx*dx
#            dy2  = dy*dy
#            dxdy = dx*dy
#            out[k] = fi[ i2_F_c ] + dx*fi[ i2_X_c ] + dy*fi[ i2_Y_c ] \
#                   + 0.5*dx2*fi[ i2_X2_c ] + dxdy*fi[ i2_XY_c ] + 0.5*dy2*fi[ i2_Y2_c ]
#
#            # symmetric Horner-like form:
#            out[k] = fi[ i2_F_c ] + dx*( fi[ i2_X_c ] + dx*( 0.5*( fi[ i2_X2_c ] ) ) ) \
#                                  + dy*( fi[ i2_Y_c ] + dy*( 0.5*( fi[ i2_Y2_c ] ) ) ) \
#
#                                  + dxdy*( fi[ i2_XY_c ] )

            resX   = fma( dx, 0.5*fi[ defs.i2_X2_c ], fi[ defs.i2_X_c ] )
            resY   = fma( dy, 0.5*fi[ defs.i2_Y2_c ], fi[ defs.i2_Y_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i2_XY_c ]
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 1:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
#            # for documentation only:
#            out[k] = fi[ i2_F_c ] + dx*fi[ i2_X_c ] + dy*fi[ i2_Y_c ]

            # sum the contributions
            #
            acc1   = dx*fi[ defs.i2_X_c ]
            acc1   = fma( dy, fi[ defs.i2_Y_c ], acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i2_F_c ]

    return 0


# Same strategy as above, but for a generic polynomial (no partially baked coefficients).
#
# Now fi[ i2_F_c ] is the constant term, fi[ i2_X ] is the coefficient of (x - xi), fi[ i2_Y ] is the coefficient of (y - yi), fi[ i2_X2 ] is the coefficient of (x - xi)**2, ...
#
cdef int evaluate_polynomial_2D( int order, double* fi, double xi, double yi, double[::view.generic,::view.contiguous] x, double* out ) nogil:
    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx, dy
    cdef double dxdy
    cdef double acc1, acc2  # accumulators
    cdef double resX, resY, resXY  # intermediate results

    if order == 4:
        for k in range(n):
            dx   = x[k,0] - xi
            dy   = x[k,1] - yi
            dxdy = dx*dy

            # symmetric Horner-like form using fused multiply-add (fma):

            # "X" terms
            acc1   = fma( dy, fi[ defs.i2_X3Y_c ], fi[ defs.i2_X3_c ] )
            acc1   = fma( dx, fi[ defs.i2_X4_c  ], acc1 )

            acc2   = fma( dy, fi[ defs.i2_X2Y_c ], fi[ defs.i2_X2_c ] )
            acc2   = fma( dx, acc1, acc2 )

            resX   = fma( dx, acc2, fi[ defs.i2_X_c ] )

            # "Y" terms
            acc1   = fma( dx, fi[ defs.i2_XY3_c ], fi[ defs.i2_Y3_c ] )
            acc1   = fma( dy, fi[ defs.i2_Y4_c  ], acc1 )

            acc2   = fma( dx, fi[ defs.i2_XY2_c ], fi[ defs.i2_Y2_c ] )
            acc2   = fma( dy, acc1, acc2 )

            resY   = fma( dy, acc2, fi[ defs.i2_Y_c ] )

            # "XY" terms
            resXY  = fma( dxdy, fi[ defs.i2_X2Y2_c ], fi[ defs.i2_XY_c ] )

            # sum the contributions
            #
            acc1   = dxdy*resXY
            acc1   = fma( dx,   resX,  acc1 )
            acc1   = fma( dy,   resY,  acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 3:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dxdy = dx*dy

            # "X" terms
            acc2   = fma( dy, fi[ defs.i2_X2Y_c ], fi[ defs.i2_X2_c ] )
            acc2   = fma( dx, fi[ defs.i2_X3_c  ], acc2 )

            resX   = fma( dx, acc2, fi[ defs.i2_X_c ] )

            # "Y" terms
            acc2   = fma( dx, fi[ defs.i2_XY2_c ], fi[ defs.i2_Y2_c ] )
            acc2   = fma( dy, fi[ defs.i2_Y3_c  ], acc2 )

            resY   = fma( dy, acc2, fi[ defs.i2_Y_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i2_XY_c ]
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 2:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
            dxdy = dx*dy

            resX   = fma( dx, fi[ defs.i2_X2_c ], fi[ defs.i2_X_c ] )
            resY   = fma( dy, fi[ defs.i2_Y2_c ], fi[ defs.i2_Y_c ] )

            # sum the contributions
            #
            acc1   = dxdy*fi[ defs.i2_XY_c ]
            acc1   = fma( dx,   resX,               acc1 )
            acc1   = fma( dy,   resY,               acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    elif order == 1:
        for k in range(n):
            dx = x[k,0] - xi
            dy = x[k,1] - yi
#            # for documentation only:
#            out[k] = fi[ i2_F_c ] + dx*fi[ i2_X_c ] + dy*fi[ i2_Y_c ]

            # sum the contributions
            #
            acc1   = dx*fi[ defs.i2_X_c ]
            acc1   = fma( dy, fi[ defs.i2_Y_c ], acc1 )
            acc1  += fi[ defs.i2_F_c ]

            out[k] = acc1

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i2_F_c ]

    return 0


# Evaluate an up to 4th order Taylor series expansion on the real axis, with its origin at (xi).
#
# The implementation uses Horner's form with fused multiply-adds.
#
# order : in, the order of the expansion (0,1,2,3 or 4)
# fi    : in, coefficient array ("order" determines the number of entries, no separate size parameter needed)
#         The ordering of the coefficients follows the numbering of the i1_* constants in wlsqm2_defs.pyx.
#         Here fi[ i1_F_c ] is f at xi, fi[ i1_X ] is df/dx at xi, fi[ i1_X2 ] is d2f/dx2 at xi, ...
# xi    : in, origin of the model
# x     : in, the points where to evaluate the Taylor expansion
# out   : out, array of size (x.shape[0],); the result
#
# Return value: 0 on success, anything else indicates failure
#
cdef int evaluate_taylor_expansion_1D( int order, double* fi, double xi, double[::view.generic] x, double* out ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx
    cdef double acc

    if order == 4:
        for k in range(n):
            dx = x[k] - xi
#            # for documentation only:
#
#            # naive form:
#            dx2 = dx*dx
#            out[k] = fi[ i1_F_c ] + dx*fi[ i1_X_c ] + 0.5*dx2*fi[ i1_X2_c ] + onesixth*dx2*dx*fi[ i1_X3_c ] + one24th*dx2*dx2*fi[ i1_X4_c ]
#
#            # Horner form:
#            out[k] = fi[ i1_F_c ] + dx*( fi[ i1_X_c ] + dx*( 0.5*fi[ i1_X2_c ] + dx*( onesixth*fi[ i1_X3_c ] + dx*( one24th*fi[ i1_X4_c ] ) ) ) )

            # Horner form using fused multiply-add:
            acc    = fma( dx, one24th*fi[ defs.i1_X4_c ], onesixth*fi[ defs.i1_X3_c ] )
            acc    = fma( dx, acc, 0.5*fi[ defs.i1_X2_c ] )
            acc    = fma( dx, acc, fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 3:
        for k in range(n):
            dx = x[k] - xi
#            # for documentation only:
#
#            # naive form:
#            dx2 = dx*dx
#            out[k] = fi[ i1_F_c ] + dx*fi[ i1_X_c ] + 0.5*dx2*fi[ i1_X2_c ] + onesixth*dx2*dx*fi[ i1_X3_c ]
#
#            # Horner form:
#            out[k] = fi[ i1_F_c ] + dx*( fi[ i1_X_c ] + dx*( 0.5*fi[ i1_X2_c ] + dx*( onesixth*fi[ i1_X3_c ] ) ) )

            # Horner form using fused multiply-add:
            acc    = fma( dx, onesixth*fi[ defs.i1_X3_c ], 0.5*fi[ defs.i1_X2_c ] )
            acc    = fma( dx, acc, fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 2:
        for k in range(n):
            dx  = x[k] - xi
#            # for documentation only:
#
#            # naive form:
#            out[k] = fi[ i1_F_c ] + dx*fi[ i1_X_c ] + 0.5*dx*dx*fi[ i1_X2_c ]
#
#            # Horner form:
#            out[k] = fi[ i1_F_c ] + dx*( fi[ i1_X_c ] + dx*( 0.5*fi[ i1_X2_c ] ) )

            # Horner form using fused multiply-add:
            acc    = fma( dx, 0.5*fi[ defs.i1_X2_c ], fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 1:
        for k in range(n):
            dx  = x[k] - xi
#            out[k] = fi[ i1_F_c ] + dx*fi[ i1_X_c ]

            # Horner form using fused multiply-add:
            out[k] = fma( dx, fi[ defs.i1_X_c ], fi[ defs.i1_F_c ] )  # fused multiply-add,  op1*op2 + op3

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i1_F_c ]

    return 0


# Same strategy as above, but for a generic polynomial (no partially baked coefficients).
#
# Now fi[ i1_F_c ] is the constant term, fi[ i1_X ] is the coefficient of (x - xi), fi[ i2_X2 ] is the coefficient of (x - xi)**2, ...
#
cdef int evaluate_polynomial_1D( int order, double* fi, double xi, double[::view.generic] x, double* out ) nogil:
    if order not in [0,1,2,3,4]:
        return -1
#        with gil:
#            raise ValueError( "order must be 1, 2, 3 or 4; got %d" % order )

    cdef int n = x.shape[0]
    cdef int k
    cdef double dx
    cdef double acc  # accumulator

    if order == 4:
        for k in range(n):
            dx = x[k] - xi

            acc    = fma( dx, fi[ defs.i1_X4_c ], fi[ defs.i1_X3_c ] )
            acc    = fma( dx, acc, fi[ defs.i1_X2_c ] )
            acc    = fma( dx, acc, fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 3:
        for k in range(n):
            dx = x[k] - xi

            acc    = fma( dx, fi[ defs.i1_X3_c ], fi[ defs.i1_X2_c ] )
            acc    = fma( dx, acc, fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 2:
        for k in range(n):
            dx = x[k] - xi

            acc    = fma( dx, fi[ defs.i1_X2_c ], fi[ defs.i1_X_c ] )
            out[k] = fma( dx, acc, fi[ defs.i1_F_c ] )

    elif order == 1:
        for k in range(n):
            dx = x[k] - xi

            out[k] = fma( dx, fi[ defs.i1_X_c ], fi[ defs.i1_F_c ] )  # fused multiply-add,  op1*op2 + op3

    else: # order == 0:
        for k in range(n):
            out[k] = fi[ defs.i1_F_c ]

    return 0

