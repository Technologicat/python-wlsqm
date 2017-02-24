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

Interpolation of fitted surrogate model.

JJ 2016-11-30
"""

from __future__ import division
from __future__ import absolute_import

import numpy as np  # needed by interpolate_fit() in Python API

cimport wlsqm.fitter.defs     as defs      # C constants
cimport wlsqm.fitter.polyeval as polyeval  # evaluation of Taylor expansions and general polynomials

####################################################
# Python API
####################################################

def interpolate_fit( xi, double[::1] fi, int dimension, int order, x, int diff=0 ):
    """def interpolate_fit( xi, double[::1] fi, int dimension, int order, x, int diff=0 ):

Interpolate the fit to given points x.

This calls the C API to interpolate the model; thus the result is guaranteed to be identical
to the internal model interpolation performed during fitting with iterative refinement.

Fused multiply-add (fma()) is used to reduce roundoff.

xi    : in, the origin of the fit; (x0,y0,z0) (double[:]) in 3D, (x0,y0) (double[:]) in 2D, or (x0) (double) in 1D
fi    : in, fit coefficients (double[::1])
order : in, degree of the surrogate polynomial (int)
x     : in, points to which to interpolate the model; x[k,:] = (x,y,z) in 3D, x[k,:] = (x,y) in 2D, or x[:] = (x) in 1D
diff  : in, one of the i1_* (if dimension==1), i2_* (if dimension==2) or i3_* (if dimension==3) constants in wlsqm.fitter.defs.

        If dimension==1:

        i1_F   : interpolate function value f (default)
        i1_X   : interpolate df/dx
        i1_X2  : interpolate d2f/dx2
        i1_X3  : ...
        i1_X4

        If dimension==2:

        i2_F   : interpolate function value f (default)
        i2_X   : interpolate df/dx
        i2_Y   : interpolate df/dy
        i2_X2  : interpolate d2f/dx2
        i2_XY  : interpolate d2f/dxdy
        i2_Y2  : interpolate d2f/dy2
        i2_X3  : ...
        i2_X2Y
        i2_XY2
        i2_Y3
        i2_X4
        i2_X3Y
        i2_X2Y2
        i2_XY3
        i2_Y4

        For dimension==3, see wlsqm.fitter.defs (there are 35 constants for the 3D case).

        Derivatives of order up to that of the model order are available.
        E.g. if order=2, then up to second-order derivatives are available.

        This can be used to obtain the derivatives at a general point
        (i.e. not only at the origin of the fit; that information
         is already contained in the coefficients, since the model
         is a polynomial).

        Note that in general, the higher the derivative order,
        the less accurate the result will be. This is due to the
        fact that the fitting error mostly concentrates into
        the higher-order terms.

Return value : rank-1 array, function value at each x.
"""
    if dimension not in [1,2,3]:
        raise ValueError( "dimension must be 1, 2 or 3; got %d" % dimension )
    if order not in [0,1,2,3,4]:
        raise ValueError( "order must be 0, 1, 2, 3 or 4; got %d" % order )

    cdef double[::1] xiManyD
    cdef double[::view.generic,::view.contiguous] xManyD

    cdef double xi1D
    cdef double[::view.generic] x1D

    cdef int nx
    x  = np.atleast_1d( x )
    nx = x.shape[0]

    cdef double[::1] out = np.empty( (nx,), dtype=np.float64 )

    cdef double nan = 0./0.
    cdef double x0, y0, z0
    if dimension >= 2:
        x0     = xi[0]
        y0     = xi[1]
        z0     = xi[2]  if dimension == 3  else  nan
        xManyD = x
        x1D    = None

    else: # dimension == 1:
        x0     = xi
        y0     = nan
        z0     = nan
        xManyD = None
        x1D    = x

    # TODO: to save a few bytes, could use "no2" (currently not saved), not "no". If order >= 1, no2 = number_of_dofs(dimension, order-1); if order == 0, the workspace fi2 is not needed.
    cdef int no = infra.number_of_dofs(dimension, order)
    cdef double[::1] fi2 = np.empty( (no,), dtype=np.float64 )

    # We don't have a Case instance here, since we are coming from the Python level.
    # Now we must create one, or at least populate the required fields manually.
    #
    cdef infra.Case case
    with nogil:
        case.dimension = dimension
        case.order     = order
        case.no        = no
        case.xi        = x0
        case.yi        = y0
        case.zi        = z0
        case.fi        = &fi[0]
        case.fi2       = &fi2[0]  # if interpolating a derivative, this work space is needed (if we had a Case object, the work space would have been created by Case_allocate())
        interpolate_nD( &case, xManyD, x1D, &out[0], diff )

    return out


def lambdify_fit( xi, fi, dimension, order, diff=0 ):  # Python only
    """def lambdify_fit( xi, fi, dimension, order, diff=0 ):

Create a Python lambda that interpolates a fitted surrogate model.

If dimension = 3, the output is lambda x,y,z : ... that interpolates the model to (x,y,z).
If dimension = 2, the output is lambda x,y   : ... that interpolates the model to (x,y).
If dimension = 1, the output is lambda x     : ... that interpolates the model to x.

The returned lambda is vectorized in the sense that it can be called with an np.array of coordinates.
In the 2D and 3D cases, the lengths of the arguments must be compatible (i.e. either all of the same length or scalars).

Note that the same xi, dimension and order must be passed here as for the model fitting.

fi[] must be the coefficients output by one of the fitting functions (see wlsqm.fitter.simple.fit_*()).

The parameter "diff" allows lambdifying a derivative of the model instead of the function itself.
See interpolate_fit() for the valid values. The default is to lambdify the function itself.
"""
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    if dimension not in [1,2,3]:
        raise ValueError( "dimension must be 1, 2 or 3; got %d" % dimension )
    if order not in [0,1,2,3,4]:
        raise ValueError( "order must be 0, 1, 2, 3 or 4; got %d" % order )

    # Make an adapter that wraps interpolate_fit(), freezing in the given xi, fi, dimension, order.
    #
    if dimension == 3:
        def model(x, y, z):
            if np.shape(y) != np.shape(x)  or  np.shape(z) != np.shape(x):
                raise ValueError("In model() (generated by wlsqm.fitter.interp.lambdify_fit()): x, y and z must be of the same shape; got shape(x) = %s, shape(y) = %s, shape(z) = %s" % (np.shape(x), np.shape(y), np.shape(z)))

            # force x and y into arrays of the same shape
            #
            nx = np.size(x)
            ny = np.size(y)
            if nx > 1  and  ny == 1:
                y = np.ones_like(x) * y
            elif nx == 1  and  ny > 1:
                x = np.ones_like(y) * x
            elif nx == 1  and  ny == 1:
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)
            # now x and y are of the same shape, and both arrays; process z
            nz = np.size(z)
            if nz == 1:
                z = np.ones_like(x) * z

            # make a single array of shape (nx,2)
            #
            # xy[j,:] contains (x,y) for the jth point
            #
            xy  = np.empty( (np.size(x),3), dtype=np.float64 )
            shp = np.shape(x)
            xy[:,0] = np.reshape(x, -1)
            xy[:,1] = np.reshape(y, -1)
            xy[:,2] = np.reshape(z, -1)

            return np.reshape( interpolate_fit( xi, fi, 3, order, xy, diff ), shp )

    elif dimension == 2:
        def model(x, y):
            if np.shape(y) != np.shape(x):
                raise ValueError("In model() (generated by wlsqm.fitter.interp.lambdify_fit()): x and y must be of the same shape; got shape(x) = %s, shape(y) = %s" % (np.shape(x), np.shape(y)))

            # force x and y into arrays of the same shape
            #
            nx = np.size(x)
            ny = np.size(y)
            if nx > 1  and  ny == 1:
                y = np.ones_like(x) * y
            elif nx == 1  and  ny > 1:
                x = np.ones_like(y) * x
            elif nx == 1  and  ny == 1:
                x = np.atleast_1d(x)
                y = np.atleast_1d(y)

            # make a single array of shape (nx,2)
            #
            # xy[j,:] contains (x,y) for the jth point
            #
            xy  = np.empty( (np.size(x),2), dtype=np.float64 )
            shp = np.shape(x)
            xy[:,0] = np.reshape(x, -1)
            xy[:,1] = np.reshape(y, -1)

            return np.reshape( interpolate_fit( xi, fi, 2, order, xy, diff ), shp )

    else: # dimension == 1:
        def model(x):
            tmp = np.atleast_1d(x)
            return np.asanyarray( interpolate_fit( xi, fi, 1, order, tmp, diff ) )  # the asanyarray() is needed to convert memoryviewslice --> np.array

    return model


####################################################
# C API
####################################################

# Interpolation of the model and its derivatives.

# Adapter that takes in parameters for 1D, 2D and 3D cases, calls the appropriate routine and passes on the relevant parameters.
#
# This is used to work around the static typing in C to allow a single dimension-agnostic implementation for functions needing model interpolation.
#
cdef int interpolate_nD( infra.Case* case, double[::view.generic,::view.contiguous] xManyD, double[::view.generic] x1D, double* out, int diff ) nogil:
    if case.dimension == 3:
        return interpolate_3D( case, xManyD, out, diff )
    elif case.dimension == 2:
        return interpolate_2D( case, xManyD, out, diff )
    else: # case.dimension == 1:
        return interpolate_1D( case, x1D, out, diff )


# case  : infra.Case object with the necessary metadata
#     xi    : in, x0 of the origin (x0,y0,z0) of the fit
#     yi    : in, y0 of the origin (x0,y0,z0) of the fit
#     zi    : in, z0 of the origin (x0,y0,z0) of the fit
#     order : in, degree of the surrogate polynomial
#     fi    : in, fit coefficients
# x     : in, x[k,:] = (x,y,z) to which to interpolate the model
# out   : out, function values. Must be allocated by caller. Must have as nk elements (one element for each point x).
# diff  : in, one of the i3_* constants in wlsqm.fitter.defs.
#
# Note that the same xi and order must be passed here as for the model fitting.
# fi[] must contain the output the model fitting.
#
cdef int interpolate_3D( infra.Case* case, double[::view.generic,::view.contiguous] x, double* out, int diff ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    cdef int order = case.order
    cdef int k,n

    # Interpolate function value? (the most common case)
    if diff == defs.i3_F_c:
        return polyeval.taylor_3D( order, case.fi, case.xi, case.yi, case.zi, x, out )

    # Interpolate a derivative of an order higher than the order of the model?
    #
    # The result is identically zero, so we special-case it here.
    #
    # Note that:
    #   - The constants used as values of "diff" are ordered in increasing order of derivatives
    #   - infra.number_of_dofs() returns one-past-end indices for each order of derivatives
    #   - This information is cached into case.no
    #
    elif diff >= case.no:  # infra.number_of_dofs( case.dimension, case.order ):
        n = x.shape[0]
        for k in range(n):
            out[k] = 0.
        return 0  # success

    # Else this derivative may be nonzero, handle it normally.

    # We fill in the coefficient array of the derivative polynomial,
    # shifting the coefficients to their new positions in the array
    # and accounting for exponents that "drop down" in the differentiation.
    #
    # We must also account for the constant factors in the Taylor expansion,
    # since fi[] contains only the DOF value data (and the DOFs are the
    # function value and derivatives at the point xi).
    #
    # Then we use the generic polynomial evaluator.

    cdef double* fi  = case.fi
    cdef double* fi2 = case.fi2  # work space

    # --------------- 1st order derivatives ---------------
    # if we get here, model order >= 1

    if diff == defs.i3_X_c:
        fi2[ defs.i3_F_c ]       = 1.*fi[ defs.i3_X_c    ]
        if order >= 2:
            fi2[ defs.i3_X_c   ] = 2.*fi[ defs.i3_X2_c   ] * 0.5
            fi2[ defs.i3_Y_c   ] = 1.*fi[ defs.i3_XY_c   ]
            fi2[ defs.i3_Z_c   ] = 1.*fi[ defs.i3_XZ_c   ]
        if order >= 3:
            fi2[ defs.i3_X2_c  ] = 3.*fi[ defs.i3_X3_c   ] * onesixth
            fi2[ defs.i3_XY_c  ] = 2.*fi[ defs.i3_X2Y_c  ] * 0.5
            fi2[ defs.i3_Y2_c  ] = 1.*fi[ defs.i3_XY2_c  ] * 0.5
            fi2[ defs.i3_YZ_c  ] = 1.*fi[ defs.i3_XYZ_c  ]
            fi2[ defs.i3_Z2_c  ] = 1.*fi[ defs.i3_XZ2_c  ] * 0.5
            fi2[ defs.i3_XZ_c  ] = 2.*fi[ defs.i3_X2Z_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X3_c  ] = 4.*fi[ defs.i3_X4_c   ] * one24th
            fi2[ defs.i3_X2Y_c ] = 3.*fi[ defs.i3_X3Y_c  ] * onesixth
            fi2[ defs.i3_XY2_c ] = 2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_Y3_c  ] = 1.*fi[ defs.i3_XY3_c  ] * onesixth
            fi2[ defs.i3_Y2Z_c ] = 1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_YZ2_c ] = 1.*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_Z3_c  ] = 1.*fi[ defs.i3_XZ3_c  ] * onesixth
            fi2[ defs.i3_XZ2_c ] = 2.*fi[ defs.i3_X2Z2_c ] * 0.25
            fi2[ defs.i3_X2Z_c ] = 3.*fi[ defs.i3_X3Z_c  ] * onesixth
            fi2[ defs.i3_XYZ_c ] = 2.*fi[ defs.i3_X2YZ_c ] * 0.5

        return polyeval.general_3D( order-1, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y_c:
        fi2[ defs.i3_F_c ]       = 1.*fi[ defs.i3_Y_c    ]
        if order >= 2:
            fi2[ defs.i3_X_c   ] = 1.*fi[ defs.i3_XY_c   ]
            fi2[ defs.i3_Y_c   ] = 2.*fi[ defs.i3_Y2_c   ] * 0.5
            fi2[ defs.i3_Z_c   ] = 1.*fi[ defs.i3_YZ_c   ]
        if order >= 3:
            fi2[ defs.i3_X2_c  ] = 1.*fi[ defs.i3_X2Y_c  ] * 0.5
            fi2[ defs.i3_XY_c  ] = 2.*fi[ defs.i3_XY2_c  ] * 0.5
            fi2[ defs.i3_Y2_c  ] = 3.*fi[ defs.i3_Y3_c   ] * onesixth
            fi2[ defs.i3_YZ_c  ] = 2.*fi[ defs.i3_Y2Z_c  ] * 0.5
            fi2[ defs.i3_Z2_c  ] = 1.*fi[ defs.i3_YZ2_c  ] * 0.5
            fi2[ defs.i3_XZ_c  ] = 1.*fi[ defs.i3_XYZ_c  ]
        if order >= 4:
            fi2[ defs.i3_X3_c  ] = 1.*fi[ defs.i3_X3Y_c  ] * onesixth
            fi2[ defs.i3_X2Y_c ] = 2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_XY2_c ] = 3.*fi[ defs.i3_XY3_c  ] * onesixth
            fi2[ defs.i3_Y3_c  ] = 4.*fi[ defs.i3_Y4_c   ] * one24th
            fi2[ defs.i3_Y2Z_c ] = 3.*fi[ defs.i3_Y3Z_c  ] * onesixth
            fi2[ defs.i3_YZ2_c ] = 2.*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_Z3_c  ] = 1.*fi[ defs.i3_YZ3_c  ] * onesixth
            fi2[ defs.i3_XZ2_c ] = 1.*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_X2Z_c ] = 1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_XYZ_c ] = 2.*fi[ defs.i3_XY2Z_c ] * 0.5

        return polyeval.general_3D( order-1, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Z_c:
        fi2[ defs.i3_F_c ]       = 1.*fi[ defs.i3_Z_c    ]
        if order >= 2:
            fi2[ defs.i3_X_c   ] = 1.*fi[ defs.i3_XZ_c   ]
            fi2[ defs.i3_Y_c   ] = 1.*fi[ defs.i3_YZ_c   ]
            fi2[ defs.i3_Z_c   ] = 2.*fi[ defs.i3_Z2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i3_X2_c  ] = 1.*fi[ defs.i3_X2Z_c  ] * 0.5
            fi2[ defs.i3_XY_c  ] = 1.*fi[ defs.i3_XYZ_c  ]
            fi2[ defs.i3_Y2_c  ] = 1.*fi[ defs.i3_Y2Z_c  ] * 0.5
            fi2[ defs.i3_YZ_c  ] = 2.*fi[ defs.i3_YZ2_c  ] * 0.5
            fi2[ defs.i3_Z2_c  ] = 3.*fi[ defs.i3_Z3_c   ] * onesixth
            fi2[ defs.i3_XZ_c  ] = 2.*fi[ defs.i3_XZ2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X3_c  ] = 1.*fi[ defs.i3_X3Z_c  ] * onesixth
            fi2[ defs.i3_X2Y_c ] = 1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_XY2_c ] = 1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_Y3_c  ] = 1.*fi[ defs.i3_Y3Z_c  ] * onesixth
            fi2[ defs.i3_Y2Z_c ] = 2.*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_YZ2_c ] = 3.*fi[ defs.i3_YZ3_c  ] * onesixth
            fi2[ defs.i3_Z3_c  ] = 4.*fi[ defs.i3_Z4_c   ] * one24th
            fi2[ defs.i3_XZ2_c ] = 3.*fi[ defs.i3_XZ3_c  ] * onesixth
            fi2[ defs.i3_X2Z_c ] = 2.*fi[ defs.i3_X2Z2_c ] * 0.25
            fi2[ defs.i3_XYZ_c ] = 2.*fi[ defs.i3_XYZ2_c ] * 0.5

        return polyeval.general_3D( order-1, fi2, case.xi, case.yi, case.zi, x, out )

    # --------------- 2nd order derivatives ---------------
    # if we get here, model order >= 2

    elif diff == defs.i3_X2_c:
        fi2[ defs.i3_F_c ]      = 1.*2.*fi[ defs.i3_X2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 2.*3.*fi[ defs.i3_X3_c   ] * onesixth
            fi2[ defs.i3_Y_c  ] = 1.*2.*fi[ defs.i3_X2Y_c  ] * 0.5
            fi2[ defs.i3_Z_c  ] = 1.*2.*fi[ defs.i3_X2Z_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 3.*4.*fi[ defs.i3_X4_c   ] * one24th
            fi2[ defs.i3_XY_c ] = 2.*3.*fi[ defs.i3_X3Y_c  ] * onesixth
            fi2[ defs.i3_Y2_c ] = 1.*2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_YZ_c ] = 1.*2.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_Z2_c ] = 1.*2.*fi[ defs.i3_X2Z2_c ] * 0.25
            fi2[ defs.i3_XZ_c ] = 2.*3.*fi[ defs.i3_X3Z_c  ] * onesixth

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XY_c:
        fi2[ defs.i3_F_c ]      = 1.*1.*fi[ defs.i3_XY_c   ]
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 2.*1.*fi[ defs.i3_X2Y_c  ] * 0.5
            fi2[ defs.i3_Y_c  ] = 1.*2.*fi[ defs.i3_XY2_c  ] * 0.5
            fi2[ defs.i3_Z_c  ] = 1.*1.*fi[ defs.i3_XYZ_c  ]
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 3.*1.*fi[ defs.i3_X3Y_c  ] * onesixth
            fi2[ defs.i3_XY_c ] = 2.*2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_Y2_c ] = 1.*3.*fi[ defs.i3_XY3_c  ] * onesixth
            fi2[ defs.i3_YZ_c ] = 1.*2.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_Z2_c ] = 1.*1.*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_XZ_c ] = 2.*1.*fi[ defs.i3_X2YZ_c ] * 0.5

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y2_c:
        fi2[ defs.i3_F_c ]      = 1.*2.*fi[ defs.i3_Y2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 1.*2.*fi[ defs.i3_XY2_c  ] * 0.5
            fi2[ defs.i3_Y_c  ] = 2.*3.*fi[ defs.i3_Y3_c   ] * onesixth
            fi2[ defs.i3_Z_c  ] = 1.*2.*fi[ defs.i3_Y2Z_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 1.*2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_XY_c ] = 2.*3.*fi[ defs.i3_XY3_c  ] * onesixth
            fi2[ defs.i3_Y2_c ] = 3.*4.*fi[ defs.i3_Y4_c   ] * one24th
            fi2[ defs.i3_YZ_c ] = 2.*3.*fi[ defs.i3_Y3Z_c  ] * onesixth
            fi2[ defs.i3_Z2_c ] = 1.*2.*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_XZ_c ] = 1.*2.*fi[ defs.i3_XY2Z_c ] * 0.5

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_YZ_c:
        fi2[ defs.i3_F_c ]      = 1.*1.*fi[ defs.i3_YZ_c   ]
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 1.*1.*fi[ defs.i3_XYZ_c  ]
            fi2[ defs.i3_Y_c  ] = 2.*1.*fi[ defs.i3_Y2Z_c  ] * 0.5
            fi2[ defs.i3_Z_c  ] = 1.*2.*fi[ defs.i3_YZ2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 1.*1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_XY_c ] = 2.*1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_Y2_c ] = 3.*1.*fi[ defs.i3_Y3Z_c  ] * onesixth
            fi2[ defs.i3_YZ_c ] = 2.*2.*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_Z2_c ] = 1.*3.*fi[ defs.i3_YZ3_c  ] * onesixth
            fi2[ defs.i3_XZ_c ] = 1.*2.*fi[ defs.i3_XYZ2_c ] * 0.5

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Z2_c:
        fi2[ defs.i3_F_c ]      = 1.*2.*fi[ defs.i3_Z2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 1.*2.*fi[ defs.i3_XZ2_c  ] * 0.5
            fi2[ defs.i3_Y_c  ] = 1.*2.*fi[ defs.i3_YZ2_c  ] * 0.5
            fi2[ defs.i3_Z_c  ] = 2.*3.*fi[ defs.i3_Z3_c   ] * onesixth
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 1.*2.*fi[ defs.i3_X2Z2_c ] * 0.25
            fi2[ defs.i3_XY_c ] = 1.*2.*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_Y2_c ] = 1.*2.*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_YZ_c ] = 2.*3.*fi[ defs.i3_YZ3_c  ] * onesixth
            fi2[ defs.i3_Z2_c ] = 3.*4.*fi[ defs.i3_Z4_c   ] * one24th
            fi2[ defs.i3_XZ_c ] = 2.*3.*fi[ defs.i3_XZ3_c  ] * onesixth

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XZ_c:
        fi2[ defs.i3_F_c ]      = 1.*1.*fi[ defs.i3_XZ_c   ]
        if order >= 3:
            fi2[ defs.i3_X_c  ] = 2.*1.*fi[ defs.i3_X2Z_c  ] * 0.5
            fi2[ defs.i3_Y_c  ] = 1.*1.*fi[ defs.i3_XYZ_c  ]
            fi2[ defs.i3_Z_c  ] = 1.*2.*fi[ defs.i3_XZ2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X2_c ] = 3.*1.*fi[ defs.i3_X3Z_c  ] * onesixth
            fi2[ defs.i3_XY_c ] = 2.*1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_Y2_c ] = 1.*1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_YZ_c ] = 1.*2.*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_Z2_c ] = 1.*3.*fi[ defs.i3_XZ3_c  ] * onesixth
            fi2[ defs.i3_XZ_c ] = 2.*2.*fi[ defs.i3_X2Z2_c ] * 0.25

        return polyeval.general_3D( order-2, fi2, case.xi, case.yi, case.zi, x, out )

    # --------------- 3rd order derivatives ---------------
    # if we get here, model order >= 3

    elif diff == defs.i3_X3_c:
        fi2[ defs.i3_F_c ]     = 1.*2.*3.*fi[ defs.i3_X3_c  ] * onesixth
        if order >= 4:
            fi2[ defs.i3_X_c ] = 2.*3.*4.*fi[ defs.i3_X4_c  ] * one24th
            fi2[ defs.i3_Y_c ] = 1.*2.*3.*fi[ defs.i3_X3Y_c ] * onesixth
            fi2[ defs.i3_Z_c ] = 1.*2.*3.*fi[ defs.i3_X3Z_c ] * onesixth

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X2Y_c:
        fi2[ defs.i3_F_c ]     = (1.*2.)*1.*fi[ defs.i3_X2Y_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = (2.*3.)*1.*fi[ defs.i3_X3Y_c  ] * onesixth
            fi2[ defs.i3_Y_c ] = (1.*2.)*2.*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_Z_c ] = (1.*2.)*1.*fi[ defs.i3_X2YZ_c ] * 0.5

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XY2_c:
        fi2[ defs.i3_F_c ]     = 1.*(1.*2.)*fi[ defs.i3_XY2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = 2.*(1.*2.)*fi[ defs.i3_X2Y2_c ] * 0.25
            fi2[ defs.i3_Y_c ] = 1.*(2.*3.)*fi[ defs.i3_XY3_c  ] * onesixth
            fi2[ defs.i3_Z_c ] = 1.*(1.*2.)*fi[ defs.i3_XY2Z_c ] * 0.5

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y3_c:
        fi2[ defs.i3_F_c ]     = 1.*2.*3.*fi[ defs.i3_Y3_c  ] * onesixth
        if order >= 4:
            fi2[ defs.i3_X_c ] = 1.*2.*3.*fi[ defs.i3_XY3_c ] * onesixth
            fi2[ defs.i3_Y_c ] = 2.*3.*4.*fi[ defs.i3_Y4_c  ] * one24th
            fi2[ defs.i3_Z_c ] = 1.*2.*3.*fi[ defs.i3_Y3Z_c ] * onesixth

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y2Z_c:
        fi2[ defs.i3_F_c ]     = (1.*2.)*1.*fi[ defs.i3_Y2Z_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = (1.*2.)*1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_Y_c ] = (2.*3.)*1.*fi[ defs.i3_Y3Z_c  ] * onesixth
            fi2[ defs.i3_Z_c ] = (1.*2.)*2.*fi[ defs.i3_Y2Z2_c ] * 0.25

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_YZ2_c:
        fi2[ defs.i3_F_c ]     = 1.*(1.*2.)*fi[ defs.i3_YZ2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = 1.*(1.*2.)*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_Y_c ] = 2.*(1.*2.)*fi[ defs.i3_Y2Z2_c ] * 0.25
            fi2[ defs.i3_Z_c ] = 1.*(2.*3.)*fi[ defs.i3_YZ3_c  ] * onesixth

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Z3_c:
        fi2[ defs.i3_F_c ]     = 1.*2.*3.*fi[ defs.i3_Z3_c  ] * onesixth
        if order >= 4:
            fi2[ defs.i3_X_c ] = 1.*2.*3.*fi[ defs.i3_XZ3_c ] * onesixth
            fi2[ defs.i3_Y_c ] = 1.*2.*3.*fi[ defs.i3_YZ3_c ] * onesixth
            fi2[ defs.i3_Z_c ] = 2.*3.*4.*fi[ defs.i3_Z4_c  ] * one24th

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XZ2_c:
        fi2[ defs.i3_F_c ]     = 1.*(1.*2.)*fi[ defs.i3_XZ2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = 2.*(1.*2.)*fi[ defs.i3_X2Z2_c ] * 0.25
            fi2[ defs.i3_Y_c ] = 1.*(1.*2.)*fi[ defs.i3_XYZ2_c ] * 0.5
            fi2[ defs.i3_Z_c ] = 1.*(2.*3.)*fi[ defs.i3_XZ3_c  ] * onesixth

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X2Z_c:
        fi2[ defs.i3_F_c ]     = (1.*2.)*1.*fi[ defs.i3_X2Z_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i3_X_c ] = (2.*3.)*1.*fi[ defs.i3_X3Z_c  ] * onesixth
            fi2[ defs.i3_Y_c ] = (1.*2.)*1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_Z_c ] = (1.*2.)*2.*fi[ defs.i3_X2Z2_c ] * 0.25

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XYZ_c:
        fi2[ defs.i3_F_c ]     = 1.*1.*1.*fi[ defs.i3_XYZ_c  ]
        if order >= 4:
            fi2[ defs.i3_X_c ] = 2.*1.*1.*fi[ defs.i3_X2YZ_c ] * 0.5
            fi2[ defs.i3_Y_c ] = 1.*2.*1.*fi[ defs.i3_XY2Z_c ] * 0.5
            fi2[ defs.i3_Z_c ] = 1.*1.*2.*fi[ defs.i3_XYZ2_c ] * 0.5

        return polyeval.general_3D( order-3, fi2, case.xi, case.yi, case.zi, x, out )

    # --------------- 4th order derivatives ---------------
    # if we get here, model order >= 4

    elif diff == defs.i3_X4_c:
        fi2[ defs.i3_F_c ] = 1.*2.*3.*4.*fi[ defs.i3_X4_c ] * one24th
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X3Y_c:
        fi2[ defs.i3_F_c ] = (1.*2.*3.)*1.*fi[ defs.i3_X3Y_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X2Y2_c:
        fi2[ defs.i3_F_c ] = 2.*2.*fi[ defs.i3_X2Y2_c ] * 0.25
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XY3_c:
        fi2[ defs.i3_F_c ] = 1.*(1.*2.*3.)*fi[ defs.i3_XY3_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y4_c:
        fi2[ defs.i3_F_c ] = 1.*2.*3.*4.*fi[ defs.i3_Y4_c ] * one24th
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y3Z_c:
        fi2[ defs.i3_F_c ] = (1.*2.*3.)*1.*fi[ defs.i3_Y3Z_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Y2Z2_c:
        fi2[ defs.i3_F_c ] = 2.*2.*fi[ defs.i3_Y2Z2_c ] * 0.25
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_YZ3_c:
        fi2[ defs.i3_F_c ] = 1.*(1.*2.*3.)*fi[ defs.i3_YZ3_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_Z4_c:
        fi2[ defs.i3_F_c ] = 1.*2.*3.*4.*fi[ defs.i3_Z4_c ] * one24th
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XZ3_c:
        fi2[ defs.i3_F_c ] = 1.*(1.*2.*3.)*fi[ defs.i3_XZ3_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X2Z2_c:
        fi2[ defs.i3_F_c ] = 2.*2.*fi[ defs.i3_X2Z2_c ] * 0.25
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X3Z_c:
        fi2[ defs.i3_F_c ] = (1.*2.*3.)*1.*fi[ defs.i3_X3Z_c ] * onesixth
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_X2YZ_c:
        fi2[ defs.i3_F_c ] = (1.*2.)*1.*1.*fi[ defs.i3_X2YZ_c ] * 0.5
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XY2Z_c:
        fi2[ defs.i3_F_c ] = 1.*(1.*2.)*1.*fi[ defs.i3_XY2Z_c ] * 0.5
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    elif diff == defs.i3_XYZ2_c:
        fi2[ defs.i3_F_c ] = 1.*1.*(1.*2.)*fi[ defs.i3_XYZ2_c ] * 0.5
        return polyeval.general_3D( order-4, fi2, case.xi, case.yi, case.zi, x, out )

    # --------------- end ---------------

    else:
        return -1  # invalid value for "diff"


# case  : infra.Case object with the necessary metadata
#     xi    : in, x0 of the origin (x0,y0) of the fit
#     yi    : in, y0 of the origin (x0,y0) of the fit
#     order : in, degree of the surrogate polynomial
#     fi    : in, fit coefficients
# x     : in, x[k,:] = (x,y) to which to interpolate the model
# out   : out, function values. Must be allocated by caller. Must have as nk elements (one element for each point x).
# diff  : in, one of the i2_* constants in wlsqm.fitter.defs.
#
# Note that the same xi and order must be passed here as for the model fitting.
# fi[] must contain the output the model fitting.
#
cdef int interpolate_2D( infra.Case* case, double[::view.generic,::view.contiguous] x, double* out, int diff ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    cdef int order = case.order
    cdef int k,n

    # Interpolate function value? (the most common case)
    if diff == defs.i2_F_c:
        return polyeval.taylor_2D( order, case.fi, case.xi, case.yi, x, out )

    # Interpolate a derivative of an order higher than the order of the model?
    #
    # The result is identically zero, so we special-case it here.
    #
    # Note that:
    #   - The constants used as values of "diff" are ordered in increasing order of derivatives
    #   - infra.number_of_dofs() returns one-past-end indices for each order of derivatives
    #   - This information is cached into case.no
    #
    elif diff >= case.no:  # infra.number_of_dofs( case.dimension, case.order ):
        n = x.shape[0]
        for k in range(n):
            out[k] = 0.
        return 0  # success

    # Else this derivative may be nonzero, handle it normally.

    # We fill in the coefficient array of the derivative polynomial,
    # shifting the coefficients to their new positions in the array
    # and accounting for exponents that "drop down" in the differentiation.
    #
    # We must also account for the constant factors in the Taylor expansion,
    # since fi[] contains only the DOF value data (and the DOFs are the
    # function value and derivatives at the point xi).
    #
    # Then we use the generic polynomial evaluator.

    cdef double* fi  = case.fi
    cdef double* fi2 = case.fi2  # work space

    # --------------- 1st order derivatives ---------------
    # if we get here, model order >= 1

    if diff == defs.i2_X_c:
        fi2[ defs.i2_F_c ]       = 1.*fi[ defs.i2_X_c    ]
        if order >= 2:
            fi2[ defs.i2_X_c   ] = 2.*fi[ defs.i2_X2_c   ] * 0.5
            fi2[ defs.i2_Y_c   ] = 1.*fi[ defs.i2_XY_c   ]
        if order >= 3:
            fi2[ defs.i2_X2_c  ] = 3.*fi[ defs.i2_X3_c   ] * onesixth
            fi2[ defs.i2_XY_c  ] = 2.*fi[ defs.i2_X2Y_c  ] * 0.5
            fi2[ defs.i2_Y2_c  ] = 1.*fi[ defs.i2_XY2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i2_X3_c  ] = 4.*fi[ defs.i2_X4_c   ] * one24th
            fi2[ defs.i2_X2Y_c ] = 3.*fi[ defs.i2_X3Y_c  ] * onesixth
            fi2[ defs.i2_XY2_c ] = 2.*fi[ defs.i2_X2Y2_c ] * 0.25
            fi2[ defs.i2_Y3_c  ] = 1.*fi[ defs.i2_XY3_c  ] * onesixth

        return polyeval.general_2D( order-1, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_Y_c:
        fi2[ defs.i2_F_c ]       = 1.*fi[ defs.i2_Y_c    ]
        if order >= 2:
            fi2[ defs.i2_X_c   ] = 1.*fi[ defs.i2_XY_c   ]
            fi2[ defs.i2_Y_c   ] = 2.*fi[ defs.i2_Y2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i2_X2_c  ] = 1.*fi[ defs.i2_X2Y_c  ] * 0.5
            fi2[ defs.i2_XY_c  ] = 2.*fi[ defs.i2_XY2_c  ] * 0.5
            fi2[ defs.i2_Y2_c  ] = 3.*fi[ defs.i2_Y3_c   ] * onesixth
        if order >= 4:
            fi2[ defs.i2_X3_c  ] = 1.*fi[ defs.i2_X3Y_c  ] * onesixth
            fi2[ defs.i2_X2Y_c ] = 2.*fi[ defs.i2_X2Y2_c ] * 0.25
            fi2[ defs.i2_XY2_c ] = 3.*fi[ defs.i2_XY3_c  ] * onesixth
            fi2[ defs.i2_Y3_c  ] = 4.*fi[ defs.i2_Y4_c   ] * one24th

        return polyeval.general_2D( order-1, fi2, case.xi, case.yi, x, out )

    # --------------- 2nd order derivatives ---------------
    # if we get here, model order >= 2

    elif diff == defs.i2_X2_c:
        fi2[ defs.i2_F_c ]      = 1.*2.*fi[ defs.i2_X2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i2_X_c  ] = 2.*3.*fi[ defs.i2_X3_c   ] * onesixth
            fi2[ defs.i2_Y_c  ] = 1.*2.*fi[ defs.i2_X2Y_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i2_X2_c ] = 3.*4.*fi[ defs.i2_X4_c   ] * one24th
            fi2[ defs.i2_XY_c ] = 2.*3.*fi[ defs.i2_X3Y_c  ] * onesixth
            fi2[ defs.i2_Y2_c ] = 1.*2.*fi[ defs.i2_X2Y2_c ] * 0.25

        return polyeval.general_2D( order-2, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_XY_c:
        fi2[ defs.i2_F_c ]      = 1.*1.*fi[ defs.i2_XY_c   ]
        if order >= 3:
            fi2[ defs.i2_X_c  ] = 2.*1.*fi[ defs.i2_X2Y_c  ] * 0.5
            fi2[ defs.i2_Y_c  ] = 1.*2.*fi[ defs.i2_XY2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i2_X2_c ] = 3.*1.*fi[ defs.i2_X3Y_c  ] * onesixth
            fi2[ defs.i2_XY_c ] = 2.*2.*fi[ defs.i2_X2Y2_c ] * 0.25
            fi2[ defs.i2_Y2_c ] = 1.*3.*fi[ defs.i2_XY3_c  ] * onesixth

        return polyeval.general_2D( order-2, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_Y2_c:
        fi2[ defs.i2_F_c ]      = 1.*2.*fi[ defs.i2_Y2_c   ] * 0.5
        if order >= 3:
            fi2[ defs.i2_X_c  ] = 1.*2.*fi[ defs.i2_XY2_c  ] * 0.5
            fi2[ defs.i2_Y_c  ] = 2.*3.*fi[ defs.i2_Y3_c   ] * onesixth
        if order >= 4:
            fi2[ defs.i2_X2_c ] = 1.*2.*fi[ defs.i2_X2Y2_c ] * 0.25
            fi2[ defs.i2_XY_c ] = 2.*3.*fi[ defs.i2_XY3_c  ] * onesixth
            fi2[ defs.i2_Y2_c ] = 3.*4.*fi[ defs.i2_Y4_c   ] * one24th

        return polyeval.general_2D( order-2, fi2, case.xi, case.yi, x, out )

    # --------------- 3rd order derivatives ---------------
    # if we get here, model order >= 3

    elif diff == defs.i2_X3_c:
        fi2[ defs.i2_F_c ]     = 1.*2.*3.*fi[ defs.i2_X3_c  ] * onesixth
        if order >= 4:
            fi2[ defs.i2_X_c ] = 2.*3.*4.*fi[ defs.i2_X4_c  ] * one24th
            fi2[ defs.i2_Y_c ] = 1.*2.*3.*fi[ defs.i2_X3Y_c ] * onesixth

        return polyeval.general_2D( order-3, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_X2Y_c:
        fi2[ defs.i2_F_c ]     = (1.*2.)*1.*fi[ defs.i2_X2Y_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i2_X_c ] = (2.*3.)*1.*fi[ defs.i2_X3Y_c  ] * onesixth
            fi2[ defs.i2_Y_c ] = (1.*2.)*2.*fi[ defs.i2_X2Y2_c ] * 0.25

        return polyeval.general_2D( order-3, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_XY2_c:
        fi2[ defs.i2_F_c ]     = 1.*(1.*2.)*fi[ defs.i2_XY2_c  ] * 0.5
        if order >= 4:
            fi2[ defs.i2_X_c ] = 2.*(1.*2.)*fi[ defs.i2_X2Y2_c ] * 0.25
            fi2[ defs.i2_Y_c ] = 1.*(2.*3.)*fi[ defs.i2_XY3_c  ] * onesixth

        return polyeval.general_2D( order-3, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_Y3_c:
        fi2[ defs.i2_F_c ]     = 1.*2.*3.*fi[ defs.i2_Y3_c  ] * onesixth
        if order >= 4:
            fi2[ defs.i2_X_c ] = 1.*2.*3.*fi[ defs.i2_XY3_c ] * onesixth
            fi2[ defs.i2_Y_c ] = 2.*3.*4.*fi[ defs.i2_Y4_c  ] * one24th

        return polyeval.general_2D( order-3, fi2, case.xi, case.yi, x, out )

    # --------------- 4th order derivatives ---------------
    # if we get here, model order >= 4

    elif diff == defs.i2_X4_c:
        fi2[ defs.i2_F_c ] = 1.*2.*3.*4.*fi[ defs.i2_X4_c ] * one24th
        return polyeval.general_2D( order-4, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_X3Y_c:
        fi2[ defs.i2_F_c ] = (1.*2.*3.)*1.*fi[ defs.i2_X3Y_c ] * onesixth
        return polyeval.general_2D( order-4, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_X2Y2_c:
        fi2[ defs.i2_F_c ] = (1.*2.)*(1.*2.)*fi[ defs.i2_X2Y2_c ] * 0.25
        return polyeval.general_2D( order-4, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_XY3_c:
        fi2[ defs.i2_F_c ] = 1.*(1.*2.*3.)*fi[ defs.i2_XY3_c ] * onesixth
        return polyeval.general_2D( order-4, fi2, case.xi, case.yi, x, out )

    elif diff == defs.i2_Y4_c:
        fi2[ defs.i2_F_c ] = 1.*2.*3.*4.*fi[ defs.i2_Y4_c ] * one24th
        return polyeval.general_2D( order-4, fi2, case.xi, case.yi, x, out )

    # --------------- end ---------------

    else:
        return -1  # invalid value for "diff"


# case  : infra.Case object with the necessary metadata
#     xi    : in, the origin (x0) of the fit
#     order : in, degree of the surrogate polynomial
#     fi    : in, fit coefficients
# x     : in, x[k] = (x) to which to interpolate the model
# out   : out, function values. Must be allocated by caller. Must have as nk elements (one element for each point x).
# diff  : in, one of the i1_* constants in wlsqm.fitter.defs.
#
# Note that the same xi and order must be passed here as for the model fitting.
# fi[] must contain the output from the model fitting.
#
cdef int interpolate_1D( infra.Case* case, double[::view.generic] x, double* out, int diff ) nogil:
    DEF onesixth = 1./6.
    DEF one24th  = 1./24.

    cdef int order = case.order
    cdef int k,n

    # Interpolate function value? (the most common case)
    if diff == defs.i1_F_c:
        return polyeval.taylor_1D( order, case.fi, case.xi, x, out )

    # Interpolate a derivative higher than order of the model?
    elif diff >= case.no:  # infra.number_of_dofs( case.dimension, case.order ):
        n = x.shape[0]
        for k in range(n):
            out[k] = 0.
        return 0  # success

    # Else this derivative may be nonzero, handle it normally.

    # We fill in the coefficient array of the derivative polynomial,
    # shifting the coefficients to their new positions in the array
    # and accounting for exponents that "drop down" in the differentiation.
    #
    # We must also account for the constant factors in the Taylor expansion,
    # since fi[] contains only the DOF value data (and the DOFs are the
    # function value and derivatives at the point xi).
    #
    # Then we use the generic polynomial evaluator.

    cdef double* fi  = case.fi
    cdef double* fi2 = case.fi2  # work space

    # --------------- 1st order derivatives ---------------
    # if we get here, model order >= 1

    if diff == defs.i1_X_c:
        fi2[ defs.i1_F_c ]      = 1.*fi[ defs.i1_X_c  ]
        if order >= 2:
            fi2[ defs.i1_X_c  ] = 2.*fi[ defs.i1_X2_c ] * 0.5
        if order >= 3:
            fi2[ defs.i1_X2_c ] = 3.*fi[ defs.i1_X3_c ] * onesixth
        if order >= 4:
            fi2[ defs.i1_X3_c ] = 4.*fi[ defs.i1_X4_c ] * one24th

        return polyeval.general_1D( order-1, fi2, case.xi, x, out )

    # --------------- 2nd order derivatives ---------------
    # if we get here, model order >= 2

    elif diff == defs.i1_X2_c:
        fi2[ defs.i1_F_c ]      = 1.*2.*fi[ defs.i1_X2_c ] * 0.5
        if order >= 3:
            fi2[ defs.i1_X_c  ] = 2.*3.*fi[ defs.i1_X3_c ] * onesixth
        if order >= 4:
            fi2[ defs.i1_X2_c ] = 3.*4.*fi[ defs.i1_X4_c ] * one24th

        return polyeval.general_1D( order-2, fi2, case.xi, x, out )

    # --------------- 3rd order derivatives ---------------
    # if we get here, model order >= 3

    elif diff == defs.i1_X3_c:
        fi2[ defs.i1_F_c ]     = 1.*2.*3.*fi[ defs.i1_X3_c ] * onesixth
        if order >= 4:
            fi2[ defs.i1_X_c ] = 2.*3.*4.*fi[ defs.i1_X4_c ] * one24th

        return polyeval.general_1D( order-3, fi2, case.xi, x, out )

    # --------------- 4th order derivatives ---------------
    # if we get here, model order >= 4

    elif diff == defs.i1_X4_c:
        fi2[ defs.i1_F_c ] = 1.*2.*3.*4.*fi[ defs.i1_X4_c ] * one24th
        return polyeval.general_1D( order-4, fi2, case.xi, x, out )

    # --------------- end ---------------

    else:
        return -1  # invalid value for "diff"

