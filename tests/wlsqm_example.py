# -*- coding: utf-8 -*-
#
"""Testing script for wlsqm, doubles as a usage example.

-JJ 2016-11-10
"""

from __future__ import division
from __future__ import absolute_import

import time

import numpy as np
import sympy as sy

import scipy.spatial  # cKDTree

import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3

try:
    import wlsqm
except ImportError:
    print "WLSQM not found; is it installed?"
    from sys import exit
    exit(1)

import sudoku_lhs


# from various scripts, e.g. miniprojects/misc/tworods/main2.py
def axis_marginize(ax, epsx, epsy):
    a = ax.axis()
    w = a[1] - a[0]
    h = a[3] - a[2]
    ax.axis( [ a[0] - w*epsx, a[1] + w*epsx, a[2] - h*epsy, a[3] + h*epsy] )


# from find_neighbors2.py
class SimpleTimer:
    def __init__(self, label="", n=None):
        self.label = label
        self.n     = n      # number of repetitions done inside the "with..." section (for averaging in timing info)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, errtype, errvalue, traceback):
        dt         = time.time() - self.t0
        identifier = ("%s" % self.label) if len(self.label) else "time taken: "
        avg        = (", avg. %gs per run" % (dt/self.n)) if self.n is not None else ""
        print "%s%gs%s" % (identifier, dt, avg)


# many simultaneous local models, 2D
#
def testmany2d():
    #########################
    # config
    #########################

    ntasks = 8  # OpenMP parallelization

    axislims = [0., 1., 0., 1.]  # [xmin, xmax, ymin, ymax], for plotting
    nvis = 201  # number of visualization points per axis

    expr    = sy.sympify("sin(pi*x) * cos(pi*y)")

    points_per_axis = 100  # for point cloud generation

    r      = 5e-2  # neighborhood radius
    max_nk = 100   # maximum number of neighbor points to accept into each neighborhood (affects memory allocation)

    knowns = 1  # function value is known

    fit_order = 4
    weighting_method = wlsqm.WEIGHT_CENTER

    max_iter = 10  # for iterative fitting method

    reps = 20  # for demonstration of solving multiple times using the same geometry

    #########################
    # the test itself
    #########################

    print
    print "=" * 79
    print "many neighborhoods, 2D case"
    print "=" * 79
    print

    # create a stratified point cloud
    print "generating sudoku sample"
    with SimpleTimer(label=("    done in ")) as s:
        S,m = sudoku_lhs.sample(2, points_per_axis, 1)
        bins_per_axis = m*points_per_axis
        S = S / float(bins_per_axis - 1)  # scale the sample from [0, bins_per_axis-1]**2 to [0, 1]**2
        npoints = len(S)
        print "    %d points" % (npoints)

    # index the point cloud for fast neighbor searching
    print "indexing sample"
    with SimpleTimer(label=("    done in ")) as s:
        tree = scipy.spatial.cKDTree( data=S )

    # If this was an IBVP, we would here get the previous state of the unknown field.
    #
    # In this example, we just sample our function f().
    #
    lambdify_numpy_2d = lambda expr: sy.lambdify(("x","y"), expr, modules="numpy")  # SymPy expr --> lambda(x,y)
    f                 = lambdify_numpy_2d(expr)
    dfdx              = lambdify_numpy_2d(sy.diff(expr, "x"))
    dfdy              = lambdify_numpy_2d(sy.diff(expr, "y"))

    print "evaluating example function"
    with SimpleTimer(label=("    done in ")) as s:
        no = wlsqm.number_of_dofs( dimension=2, order=fit_order )
        fi = np.empty( (npoints,no), dtype=np.float64 )
        fi[:,0] = f( S[:,0], S[:,1] )  # fi[i,0] contains the function value at point S[i,:]

    # find the neighborhoods
    print "generating neighborhoods for each point"
    with SimpleTimer(label=("    done in ")) as s:
        hoods = np.zeros( (npoints,max_nk), dtype=np.int32 )  # neighbor point indices (pointing to rows in S[])
        nk    = np.empty( (npoints,), dtype=np.int32 )        # number of neighbors, i.e. nk[i] is the number of actually used columns in hoods[i,:]
        for i in xrange(npoints):
            I = tree.query_ball_point( S[i], r )  # indices of neighbors of S[i] at distance <= r  (but also including S[i] itself!)
            I = filter( lambda idx: idx != i, I )  # exclude S[i] itself
            if len(I) > max_nk:
                I = I[:max_nk]
            I = np.array( I, dtype=np.int32 )
            nk[i] = len(I)
            hoods[i,:nk[i]] = I

    # DEBUG
    print "number of neighbors min = %g, avg = %g, max = %g" % ( np.min(nk), np.mean(nk), np.max(nk) )
    print "neighbor lists for each problem instance:"
    print hoods
    print "number of neighbors for each problem instance:"
    print nk

    # perform the fitting
    print "fitting %d local surrogate models of order %d, driver mode (fit each model once)" % (npoints, fit_order)
    fit_order_array = fit_order        * np.ones( (npoints,), dtype=np.int32 )
    knowns_array    = knowns           * np.ones( (npoints,), dtype=np.int64 )
    wm_array        = weighting_method * np.ones( (npoints,), dtype=np.int32 )
    with SimpleTimer(label=("    done in ")) as s:
#        max_iterations_taken = wlsqm.fit_2D_many( xk=S[hoods], fk=fi[hoods,0], nk=nk,
#                                                   xi=S, fi=fi,
#                                                   sens=None, do_sens=False,
#                                                   order=fit_order_array, knowns=knowns_array, weighting_method=wm_array,
#                                                   debug=False )
        max_iterations_taken = wlsqm.fit_2D_many_parallel( xk=S[hoods], fk=fi[hoods,0], nk=nk,
                                                            xi=S, fi=fi,
                                                            sens=None, do_sens=False,
                                                            order=fit_order_array, knowns=knowns_array, weighting_method=wm_array,
                                                            ntasks=ntasks, debug=False )
#        max_iterations_taken = wlsqm.fit_2D_iterative_many( xk=S[hoods], fk=fi[hoods,0], nk=nk,
#                                                             xi=S, fi=fi,
#                                                             sens=None, do_sens=False,
#                                                             order=fit_order_array, knowns=knowns_array, weighting_method=wm_array,
#                                                             max_iter=max_iter, debug=False )
#        max_iterations_taken = wlsqm.fit_2D_iterative_many_parallel( xk=S[hoods], fk=fi[hoods,0], nk=nk,
#                                                            xi=S, fi=fi,
#                                                            sens=None, do_sens=False,
#                                                            order=fit_order_array, knowns=knowns_array, weighting_method=wm_array,
#                                                            max_iter=max_iter, ntasks=ntasks, debug=False )

    # Expert mode: allows solving multiple times (with new fk data) in the same geometry, performing the prepare step only once.
    #
    # This is especially good for a large number of repetitions with ALGO_BASIC, where a large majority of the computational cost comes from the prepare step.
    #
    # The total advantage is slightly smaller for a small number of repetitions with ALGO_ITERATIVE,
    # since the iterative mode already uses this strategy internally (also when invoked in driver mode).
    #
    print "fitting %d local surrogate models of order %d, expert mode" % (npoints, fit_order)
    print "    init"
    with SimpleTimer(label=("        done in ")) as s:
        solver = wlsqm.ExpertSolver( dimension=2, nk=nk, order=fit_order_array, knowns=knowns_array, weighting_method=wm_array, algorithm=wlsqm.ALGO_BASIC, do_sens=False, max_iter=max_iter, ntasks=ntasks, debug=False )
    print "    prepare"
    with SimpleTimer(label=("        done in ")) as s:
        solver.prepare( xi=S, xk=S[hoods] )
    print "    fit (each model %d times)" % (reps)
    with SimpleTimer(label=("        %d reps done in " % reps), n=reps) as s:
        for k in xrange(reps):
            solver.solve( fk=fi[hoods,0], fi=fi, sens=None )

    # DEBUG
    print "max_iterations_taken: %d" % (max_iterations_taken)
    # see that we got the derivatives at each point
    if fit_order > 0:  # no derivatives if piecewise constant fit
        print dfdx( S[:,0], S[:,1] ) - fi[:,1]
        print dfdy( S[:,0], S[:,1] ) - fi[:,2]

    #########################
    # plotting
    #########################

    xx   = np.linspace(axislims[0], axislims[1], nvis)
    yy   = np.linspace(axislims[2], axislims[3], nvis)
    X,Y  = np.meshgrid(xx, yy)
    W    = f(X,Y)

    shp = np.shape(X)
    Xlin = np.reshape(X, -1)
    Ylin = np.reshape(Y, -1)
    x = np.empty( (len(Xlin), 2), dtype=np.float64 )
    x[:,0] = Xlin
    x[:,1] = Ylin
    print "preparing to interpolate global model"
    with SimpleTimer(label=("    done in ")) as s:
        solver.prep_interpolate()
    print "interpolating global model to %d points" % (len(Xlin))
    with SimpleTimer(label=("    done in ")) as s:
        W2,dummy = solver.interpolate( x, mode='continuous', r=r )  # slow, continuous
#        W2,dummy = solver.interpolate( x, mode='nearest' )  # fast, surprisingly accurate if a reasonable number of points (and continuous-looking although technically has jumps over Voronoi cell boundaries)
    W2 = np.reshape( W2, shp )

    # make 3d plot of the function
    #
    # see http://matplotlib.sourceforge.net/examples/mplot3d/lines3d_demo.html

    fig = pl.figure(3, figsize=(12,12))
    pl.clf()

    # Axes3D has a tendency to underestimate how much space it needs; it draws its labels
    # outside the window area in certain orientations.
    #
    # This causes the labels to be clipped, which looks bad. We prevent this by creating the axes
    # in a slightly smaller rect (leaving a margin). This way the labels will show - outside the Axes3D,
    # but still inside the figure window.
    #
    # The final touch is to set the window background to a matching white, so that the
    # background of the figure appears uniform.
    #
    fig.patch.set_color( (1,1,1) )
    fig.patch.set_alpha( 1.0 )
    x0y0wh = [ 0.02, 0.02, 0.96, 0.96 ]  # left, bottom, width, height      (here as fraction of subplot area)

#    # compute the corresponding figure coordinates for the 2x1 subplot layout
#    x0y0wh[0] = 0.5 + 0.5*x0y0wh[0]  # left
#    x0y0wh[2] = 0.5*x0y0wh[2]        # width

    ax = p3.Axes3D(fig, rect=x0y0wh)

    stride = max(1, (nvis-1)//10)  # pick a good-looking stride (for lines; we actually have more vertices, making a smoother-looking curve between the lines)
    # use linewidth=0 to remove the wireframe if desired.
#    surf = ax.plot_surface(X,Y,W, rstride=stride, cstride=stride, cmap=matplotlib.cm.Blues_r, clim=[fmin,fmax], linewidth=0.25, alpha=0.5)
    ax.plot_wireframe(X,Y,W, rstride=stride, cstride=stride, color='k', linewidth=0.5, linestyle='solid')
#        pl.colorbar(surf, shrink=0.5, aspect=5)
#        pl.colorbar(surf, shrink=0.96)

    # sampled points
    if points_per_axis < 50:
        ax.plot( S[:,0], S[:,1], f( S[:,0], S[:,1] ), linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )  # exact

    # surrogate model (global, patched)
    ax.plot_wireframe(X,Y,W2, rstride=stride, cstride=stride, color='r', linewidth=0.5, linestyle='solid')

#    ax.view_init(20, -48)
#    ax.view_init(18, -46)
#    ax.view_init(18, -128)
    ax.view_init(34, 140)

    ax.axis('tight')
    ax.set_zlim(-1.01, 1.01)
    pl.xlabel('$x$')
    pl.ylabel('$y$')
    ax.set_title('f(x,y)')


    print "    uninit"
    with SimpleTimer(label=("        done in ")) as s:
        del solver


# one local model, 3D
#
def test3d():
    #########################
    # config
    #########################

    axislims = [0., 1., 0., 1.]  # [xmin, xmax, ymin, ymax], for plotting
    nvis = 101  # number of visualization points per axis

    # Let's manufacture a solution (for which we know the derivatives analytically):
    #
    expr    = sy.sympify("sin(pi*x) * cos(pi*y) * exp(z)")
#    expr    = sy.sympify("exp(x)*exp(y)*exp(z)")
#    expr    = sy.sympify("1*x + 2*y + 3*z")
#    expr    = sy.sympify("0 + 1*x + 2*y + 3*z + 4*x**2 + 5*x*y + 6*y**2 + 7*y*z + 8*z**2 + 9*x*z + 10*x**3 + 11*x**2*y + 12*x*y**2 + 13*y**3 + 14*y**2*z + 15*y*z**2 + 16*z**3 + 17*z**2*x + 18*z*x**2 + 19*x*y*z")

    noise_eps = 0#1e-3 # introduce this much Gaussian noise into each sampled function value (use 0. to turn off)

    xi = np.array( (0.45, 0.25, 0.35) )  # point (x,y,z) where we wish to find the derivatives
#    xi = np.array( (0., 0., 0.) )  # point (x,y,z) where we wish to find the derivatives

    # Degree of the surrogate polynomial; a full polynomial of this order will be used.
    #
    # In the fit, when compared to the original function (if any is available),
    # usually the highest order will be nonsense, and the lower orders will be pretty accurate.
    #
    # (I.e. the unfittable part seems to favor the highest order; which OTOH has the highest spatial frequency. Maybe there's something here?)
    #
    fit_order = 4  # 0 (constant), 1 (linear), 2 (quadratic), 3 (cubic) or 4 (quartic)

#    weighting_method = wlsqm.WEIGHT_UNIFORM  # best overall fit for function values
    weighting_method = wlsqm.WEIGHT_CENTER  # emphasize center to improve derivatives at the point xi

    max_iter = 100  # maximum number of refinement iterations for iterative fitting

    do_sens = False  # do sensitivity analysis of solution? ( d( fi[j] ) / d( fk[k] ) )

    debug = False#True  # print row scaling and condition number information? (if True, then do_sens must be False; the combination with both True is not supported)

    # Bitmask of what we know at point xi. In this example, just set the bits;
    # the data (from expr) will be automatically inserted into fi[].
    #
    # See the constants b3_* in wlsqm.fitter.defs.
    #
    knowns = 1

    # How many neighbor points to generate (to simulate the meshless 'grid').
    #
    # At least n_unknowns  points are needed to make the model fitting work at all
    # (but then the fit will be nonsensical, since it is possible to make the polynomial
    #  pass through exactly those points).
    #
    # n_unknows + 1  is the first value that makes the fitting overdetermined,
    # i.e. where the least-squares procedure starts providing any advantage.
    #
    # Here "unknown" means any element of fi[] not tagged as known in the "knowns" bitmask.
    #
    nk = 200  # used if grid_type == 'random'

    r  = 1e-1  # neighborhood radius

#    grid_type = 'random'
    grid_type = 'stencil'
#    grid_type = 'sudoku'

    #########################
    # the test itself
    #########################

    print
    print "=" * 79
    print "3D case"
    print "=" * 79
    print

    print "expr: %s, xi = %s" % (expr, xi)

    labels = ["F",
              "DX", "DY", "DZ",
              "DX2", "DXDY", "DY2", "DYDZ", "DZ2", "DXDZ",
              "DX3", "DX2DY", "DXDY2", "DY3", "DY2DZ", "DYDZ2", "DZ3", "DXDZ2", "DX2DZ", "DXDYDZ",
              "DX4", "DX3DY", "DX2DY2", "DXDY3", "DY4", "DY3DZ", "DY2DZ2", "DYDZ3", "DZ4", "DXDZ3", "DX2DZ2", "DX3DZ", "DX2DYDZ", "DXDY2DZ", "DXDYDZ2" ]
    print "legend: %s" % ("\t".join(labels))
    knowns_str = ""
    for j in range(wlsqm.SIZE3):  # SIZE3 = maximum size of c matrix for 3D case
        if j > 0:
            knowns_str += '\t'
        if knowns & (1 << j):
            knowns_str += labels[j]
    print "knowns: %s" % knowns_str
#    # http://stackoverflow.com/questions/699866/python-int-to-binary
#    print "knowns (mask): %s" % format(knowns, '010b')[::-1]

    print "surrogate order: %d" % fit_order

    if noise_eps > 0.:
        print "simulating noisy input with eps = %g" % noise_eps

    # SymPy expr --> lambda(x,y)
    lambdify_numpy_3d = lambda expr: sy.lambdify(("x","y","z"), expr, modules="numpy")
    f          = lambdify_numpy_3d(expr)

    dfdx       = lambdify_numpy_3d(sy.diff(expr, "x"))
    dfdy       = lambdify_numpy_3d(sy.diff(expr, "y"))
    dfdz       = lambdify_numpy_3d(sy.diff(expr, "z"))

    d2fdx2     = lambdify_numpy_3d(sy.diff(expr, "x", 2))
    d2fdxdy    = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "y" ))
    d2fdy2     = lambdify_numpy_3d(sy.diff(expr, "y", 2))
    d2fdydz    = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y"), "z" ))
    d2fdz2     = lambdify_numpy_3d(sy.diff(expr, "z", 2))
    d2fdxdz    = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "z" ))

    d3fdx3     = lambdify_numpy_3d(sy.diff(expr, "x", 3))
    d3fdx2dy   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 2), "y" ))
    d3fdxdy2   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "y", 2 ))
    d3fdy3     = lambdify_numpy_3d(sy.diff(expr, "y", 3))
    d3fdy2dz   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y", 2), "z" ))
    d3fdydz2   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y"), "z", 2 ))
    d3fdz3     = lambdify_numpy_3d(sy.diff(expr, "z", 3))
    d3fdxdz2   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "z", 2 ))
    d3fdx2dz   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 2), "z" ))
    d3fdxdydz  = lambdify_numpy_3d(sy.diff( sy.diff( sy.diff(expr, "x"), "y"), "z"))

    d4fdx4     = lambdify_numpy_3d(sy.diff(expr, "x", 4))
    d4fdx3dy   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 3), "y" ))
    d4fdx2dy2  = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 2), "y", 2 ))
    d4fdxdy3   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "y", 3 ))
    d4fdy4     = lambdify_numpy_3d(sy.diff(expr, "y", 4))
    d4fdy3dz   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y", 3), "z" ))
    d4fdy2dz2  = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y", 2), "z" , 2))
    d4fdydz3   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "y"), "z", 3 ))
    d4fdz4     = lambdify_numpy_3d(sy.diff(expr, "z", 4))
    d4fdxdz3   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x"), "z", 3 ))
    d4fdx2dz2  = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 2), "z" , 2))
    d4fdx3dz   = lambdify_numpy_3d(sy.diff( sy.diff(expr, "x", 3), "z" ))
    d4fdx2dydz = lambdify_numpy_3d(sy.diff( sy.diff( sy.diff(expr, "x", 2), "y"), "z"))
    d4fdxdy2dz = lambdify_numpy_3d(sy.diff( sy.diff( sy.diff(expr, "x"), "y", 2), "z"))
    d4fdxdydz2 = lambdify_numpy_3d(sy.diff( sy.diff( sy.diff(expr, "x"), "y"), "z", 2))

    # list so we can refer to the functions by indices
    funcs = ( f,
              dfdx, dfdy, dfdz,
              d2fdx2, d2fdxdy, d2fdy2, d2fdydz, d2fdz2, d2fdxdz,
              d3fdx3, d3fdx2dy, d3fdxdy2, d3fdy3, d3fdy2dz, d3fdydz2, d3fdz3, d3fdxdz2, d3fdx2dz, d3fdxdydz,
              d4fdx4, d4fdx3dy, d4fdx2dy2, d4fdxdy3, d4fdy4, d4fdy3dz, d4fdy2dz2, d4fdydz3, d4fdz4, d4fdxdz3, d4fdx2dz2, d4fdx3dz, d4fdx2dydz, d4fdxdy2dz, d4fdxdydz2
            )

    # create neighbor points xk around the point xi - this simulates our meshless 'grid'
    #
    if grid_type == 'random':
        xk = np.tile(xi, (nk,1)) + r*2.*( np.random.sample( (nk,3) ) - 0.5 )

    elif grid_type == 'stencil':
        points_per_axis = max(1,fit_order) + 1

        tt = np.linspace(-1., 1., points_per_axis)
        X,Y,Z = np.meshgrid(tt,tt,tt)
        X = np.reshape(X, -1)
        Y = np.reshape(Y, -1)
        Z = np.reshape(Z, -1)

        # convert to list of (x,y) pairs, rejecting the point (0,0) (that represents xi itself), if present
        point_list = [ (x,y,z) for x,y,z in zip(X,Y,Z) if (x,y,z) != (0.,0.,0.) ]

        nk = len(point_list)
        xk = np.array( [ ( xi[0] + r*p[0], xi[1] + r*p[1], xi[2] + r*p[2] ) for p in point_list ] )

    elif grid_type == 'sudoku':
        points_per_axis = max(1,fit_order) + 1

        S,m = sudoku_lhs.sample(3, points_per_axis, 1)
        bins_per_axis = points_per_axis*m
        S = S / float(bins_per_axis - 1)  # scale the sample from [0, bins_per_axis-1]**3 to [0, 1]**3
        S = 2. * (S - 0.5) # move to [-1, 1]**3

        # If points_per_axis is odd, a bin exists exactly at the center, so Sudoku LHS may place one point there.
        #
        # This would coincide with the point xi, so it is not useful, because we want the neighbors to be
        # distinct from xi.
        #
        # Thus, for odd points_per_axis, filter the sample to remove the point at the origin if it happens to be there.
        # Note that because of the scaling, the coordinates might not be exactly zero. We HACK by checking numerical equality;
        # a proper solution would be to filter S before the conversion to float.
        #
        if points_per_axis % 2 == 1:
            point_list = S.tolist()
            oldlen = len(point_list)
            point_list = filter( lambda item: not (abs(item[0]) < 1e-8 and abs(item[1]) < 1e-8 and abs(item[2]) < 1e-8), point_list )
            S = np.array(point_list)
            if len(point_list) < oldlen:
                print "Sudoku LHS sampled the point at the origin; discarding it from the sample"

        nk = len(S)
        xk = np.tile(xi, (nk,1)) + r*S

    else:
        raise ValueError("Unknown grid_type '%s'; valid: 'random', 'stencil', 'sudoku'" % grid_type)

    # sample the function values at the neighbor points xk (these are used to fit the surrogate model)
    #
    sample_also_xi_str = " (and xi itself)" if knowns & 1 else ""
    print "sampling %d points%s" % (nk, sample_also_xi_str)
    fk = np.empty( (nk,), dtype=np.float64 )
    for k in xrange(nk):
        fk[k] = f( xk[k,0], xk[k,1], xk[k,2] )

    # simulate numerical errors by adding noise to the neighbor point function value samples
    #
    if noise_eps > 0.:
#        # uniform
#        noise = noise_eps*2.*(np.random.sample( np.shape(fk) ) - 0.5)

        # Gaussian, truncated
        mu    = 0.0
        sigma = noise_eps / 3.
        noise = np.random.normal( loc=mu, scale=sigma, size=np.shape(fk) )
        noise[noise < -3.*sigma] = -3.*sigma
        noise[noise > +3.*sigma] = +3.*sigma

        fk += noise

    # set knowns *at point xi*
    #
    # we use nan to spot unfilled entries
    fi = np.nan * np.empty( (wlsqm.SIZE3,), dtype=np.float64 )  # F, DX, DY, DZ, ... at point xi
    for d in range(wlsqm.SIZE3):
        if knowns & (1 << d):
            fi[d] = funcs[d]( xi[0], xi[1], xi[2] )  # fill in the known value  # TODO: add noise here too?

    # allocate array for sensitivity data
    #
    # for output; sens[k,j] = d(fi[j])/d(fk[k]) if f[i] unknown
    #                         nan if fi[j] known
    #
    # Note that if order=1, the part on second derivatives is not touched (so that an (nk,3) array
    # is valid); hence we pre-fill by nan.
    #
    if do_sens:
        sens = np.nan * np.empty( (nk,wlsqm.SIZE3), dtype=np.float64 )
    else:
        sens = None

    # fit the surrogate model (see wlsqm.fitter.simple for detailed documentation)
    #
    if debug:
        print  # blank line before debug info

    iterations_taken = wlsqm.fit_3D_iterative( xk, fk, xi, fi, sens, do_sens=do_sens, order=fit_order, knowns=knowns, debug=debug, weighting_method=weighting_method, max_iter=max_iter )
#    iterations_taken = wlsqm.fit_3D( xk, fk, xi, fi, sens, do_sens=do_sens, order=fit_order, knowns=knowns, debug=debug, weighting_method=weighting_method )

    print "refinement iterations taken: %d" % iterations_taken

    # check exact solution and relative error
    #
    exact = np.array( map( lambda func : func( xi[0], xi[1], xi[2] ), funcs ) )
    err   = (fi - exact)

    print
    print "derivatives at xi:"
    print "exact:"
    print exact
    print "wlsqm solution:"
    print fi
    if do_sens:
        print "sensitivity:"
        print sens
    print "abs error:"
    print err
    print "rel error:"
    print (err / exact)

    #########################
    # plotting
    #########################

    # surrogate model - the returned fi[] are actually the coefficients of a polynomial
    model    = wlsqm.lambdify_fit( xi, fi, dimension=3, order=fit_order )  # lambda x,y : ...

    print
    print "function values at neighbor points:"
    fxk =     f( xk[:,0], xk[:,1], xk[:,2] )
    mxk = model( xk[:,0], xk[:,1], xk[:,2] )
    print "exact:"
    print fxk
    print "wlsqm solution:"
    print mxk
    print "abs error:"
    errf = mxk - fxk
    print errf
    print "rel error:"
    print (errf / fxk)

    # comparison
    xx2        = np.linspace(xi[0] - r, xi[0] + r, nvis)
    yy2        = np.linspace(xi[1] - r, xi[1] + r, nvis)
    zz2        = np.linspace(xi[2] - r, xi[2] + r, nvis)
    X2,Y2,Z2   = np.meshgrid(xx2, yy2, zz2)
    W2         = model(X2,Y2,Z2)
    W3         =     f(X2,Y2,Z2)
    diff       = W2 - W3  # fitted - exact
    idx        = np.argmax(np.abs( diff ))
    diff_lin   = np.reshape(diff, -1)
    W3_lin     = np.reshape(W3, -1)
    maxerr_abs = diff_lin[idx]
    maxerr_rel = diff_lin[idx] / W3_lin[idx]
    print "largest absolute total fit error (over the domain of the fit, not just the neighbor points):"
    print "absolute: %g" % (maxerr_abs)
    print "relative: %g" % (maxerr_rel)


# one local model, 2D
#
def test2d():
    #########################
    # config
    #########################

    axislims = [0., 1., 0., 1.]  # [xmin, xmax, ymin, ymax], for plotting
    nvis = 101  # number of visualization points per axis

    # Let's manufacture a solution (for which we know the derivatives analytically):
    #
#    expr     = sy.sympify("2*x + 3*y")
#    expr     = sy.sympify("0.2*x + 0.3*y")
#    expr     = sy.sympify("1.0 + 2*x + 3*y + 4*x**2 + 5*x*y + 6*y**2")
#    expr     = sy.sympify("0.1 + 0.2*x + 0.3*y + 0.4*x**2 + 0.5*x*y + 0.6*y**2")
#    expr    = sy.sympify("sin(pi*x)")
    expr    = sy.sympify("sin(pi*x) * cos(pi*y)")
#    expr    = sy.sympify("exp(x) * 1/(1 + y) - 1")
#    expr    = sy.sympify("exp(x) * log(1 + y)")
#    expr     = sy.sympify("1.0 + 2*x + 3*y + 4*x**2 + 5*x*y + 6*y**2 + 7*x**3 + 8*y**4")

    noise_eps = 0#1e-3 # introduce this much Gaussian noise into each sampled function value (use 0. to turn off)

    xi = np.array( (0.45, 0.25) )  # point (x,y) where we wish to find the derivatives

    # Degree of the surrogate polynomial; a full polynomial of this order will be used.
    #
    # In the fit, when compared to the original function (if any is available),
    # usually the highest order will be nonsense, and the lower orders will be pretty accurate.
    #
    # (I.e. the unfittable part seems to favor the highest order; which OTOH has the highest spatial frequency. Maybe there's something here?)
    #
    fit_order = 4  # 0 (constant), 1 (linear), 2 (quadratic), 3 (cubic) or 4 (quartic)

#    weighting_method = wlsqm.WEIGHT_UNIFORM  # best overall fit for function values
    weighting_method = wlsqm.WEIGHT_CENTER  # emphasize center to improve derivatives at the point xi

    max_iter = 100  # maximum number of refinement iterations for iterative fitting

    do_sens = False  # do sensitivity analysis of solution? ( d( fi[j] ) / d( fk[k] ) )

    debug = False#True  # print row scaling and condition number information? (if True, then do_sens must be False; the combination with both True is not supported)

    # Bitmask of what we know at point xi. In this example, just set the bits;
    # the data (from expr) will be automatically inserted into fi[].
    #
    # Bits from least sig. to most sig.: F, DX, DY, DX2, DXDY, DY2, ... (see ordering of "labels", below)
    #
    knowns = 1

    # How many neighbor points to generate (to simulate the meshless 'grid').
    #
    # At least n_unknowns  points are needed to make the model fitting work at all
    # (but then the fit will be nonsensical, since it is possible to make the polynomial
    #  pass through exactly those points).
    #
    # n_unknows + 1  is the first value that makes the fitting overdetermined,
    # i.e. where the least-squares procedure starts providing any advantage.
    #
    # Here "unknown" means any element of fi[] not tagged as known in the "knowns" bitmask.
    #
    nk = 24  # used if grid_type == 'random'

    r  = 1e-1  # neighborhood radius

#    grid_type = 'random'
#    grid_type = 'stencil'
    grid_type = 'sudoku'

    #########################
    # the test itself
    #########################

    print
    print "=" * 79
    print "2D case"
    print "=" * 79
    print

    print "expr: %s, xi = %s" % (expr, xi)

    labels = ["F", "DX", "DY", "DX2", "DXDY", "DY2", "DX3", "DX2DY", "DXDY2", "DY3", "DX4", "DX3DY", "DX2DY2", "DXDY3", "DY4"]
    print "legend: %s" % ("\t".join(labels))
    knowns_str = ""
    for j in range(wlsqm.SIZE2):  # SIZE2 = maximum size of c matrix for 2D case
        if j > 0:
            knowns_str += '\t'
        if knowns & (1 << j):
            knowns_str += labels[j]
    print "knowns: %s" % knowns_str
#    # http://stackoverflow.com/questions/699866/python-int-to-binary
#    print "knowns (mask): %s" % format(knowns, '010b')[::-1]

    print "surrogate order: %d" % fit_order

    if noise_eps > 0.:
        print "simulating noisy input with eps = %g" % noise_eps

    # SymPy expr --> lambda(x,y)
    lambdify_numpy_2d = lambda expr: sy.lambdify(("x","y"), expr, modules="numpy")
    f         = lambdify_numpy_2d(expr)
    dfdx      = lambdify_numpy_2d(sy.diff(expr, "x"))
    dfdy      = lambdify_numpy_2d(sy.diff(expr, "y"))
    d2fdx2    = lambdify_numpy_2d(sy.diff(expr, "x", 2))
    d2fdxdy   = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x"), "y" ))
    d2fdy2    = lambdify_numpy_2d(sy.diff(expr, "y", 2))
    d3fdx3    = lambdify_numpy_2d(sy.diff(expr, "x", 3))
    d3fdx2dy  = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x", 2), "y" ))
    d3fdxdy2  = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x"), "y", 2 ))
    d3fdy3    = lambdify_numpy_2d(sy.diff(expr, "y", 3))
    d4fdx4    = lambdify_numpy_2d(sy.diff(expr, "x", 4))
    d4fdx3dy  = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x", 3), "y" ))
    d4fdx2dy2 = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x", 2), "y", 2 ))
    d4fdxdy3  = lambdify_numpy_2d(sy.diff( sy.diff(expr, "x"), "y", 3 ))
    d4fdy4    = lambdify_numpy_2d(sy.diff(expr, "y", 4))
    funcs = (f, dfdx, dfdy, d2fdx2, d2fdxdy, d2fdy2, d3fdx3, d3fdx2dy, d3fdxdy2, d3fdy3, d4fdx4, d4fdx3dy, d4fdx2dy2, d4fdxdy3, d4fdy4)  # list so we can refer to the functions by indices

    # create neighbor points xk around the point xi - this simulates our meshless 'grid'
    #
    if grid_type == 'random':
        xk = np.tile(xi, (nk,1)) + r*2.*( np.random.sample( (nk,2) ) - 0.5 )

    elif grid_type == 'stencil':
        points_per_axis = max(1,fit_order) + 1

        tt = np.linspace(-1., 1., points_per_axis)
        X,Y = np.meshgrid(tt,tt)
        X = np.reshape(X, -1)
        Y = np.reshape(Y, -1)

        # convert to list of (x,y) pairs, rejecting the point (0,0) (that represents xi itself), if present
        point_list = [ (x,y) for x,y in zip(X,Y) if (x,y) != (0.,0.) ]

        nk = len(point_list)
        xk = np.array( [ ( xi[0] + r*p[0], xi[1] + r*p[1] ) for p in point_list ] )

    elif grid_type == 'sudoku':
        points_per_axis = max(1,fit_order) + 1

        S,m = sudoku_lhs.sample(2, points_per_axis, 1)
        bins_per_axis = points_per_axis*m
        S = S / float(bins_per_axis - 1)  # scale the sample from [0, bins_per_axis-1]**2 to [0, 1]**2
        S = 2. * (S - 0.5) # move to [-1, 1]**2

        # If points_per_axis is odd, a bin exists exactly at the center, so Sudoku LHS may place one point there.
        #
        # This would coincide with the point xi, so it is not useful, because we want the neighbors to be
        # distinct from xi.
        #
        # Thus, for odd points_per_axis, filter the sample to remove the point at the origin if it happens to be there.
        # Note that because of the scaling, the coordinates might not be exactly zero. We HACK by checking numerical equality;
        # a proper solution would be to filter S before the conversion to float.
        #
        if points_per_axis % 2 == 1:
            point_list = S.tolist()
            oldlen = len(point_list)
            point_list = filter( lambda item: not (abs(item[0]) < 1e-8 and abs(item[1]) < 1e-8), point_list )
            S = np.array(point_list)
            if len(point_list) < oldlen:
                print "Sudoku LHS sampled the point at the origin; discarding it from the sample"

        nk = len(S)
        xk = np.tile(xi, (nk,1)) + r*S

    else:
        raise ValueError("Unknown grid_type '%s'; valid: 'random', 'stencil', 'sudoku'" % grid_type)

    # sample the function values at the neighbor points xk (these are used to fit the surrogate model)
    #
    sample_also_xi_str = " (and xi itself)" if knowns & 1 else ""
    print "sampling %d points%s" % (nk, sample_also_xi_str)
    fk = np.empty( (nk,), dtype=np.float64 )
    for k in xrange(nk):
        fk[k] = f( xk[k,0], xk[k,1] )

    # simulate numerical errors by adding noise to the neighbor point function value samples
    #
    if noise_eps > 0.:
#        # uniform
#        noise = noise_eps*2.*(np.random.sample( np.shape(fk) ) - 0.5)

        # Gaussian, truncated
        mu    = 0.0
        sigma = noise_eps / 3.
        noise = np.random.normal( loc=mu, scale=sigma, size=np.shape(fk) )
        noise[noise < -3.*sigma] = -3.*sigma
        noise[noise > +3.*sigma] = +3.*sigma

        fk += noise

    # set knowns *at point xi*
    #
    # we use nan to spot unfilled entries
    fi = np.nan * np.empty( (wlsqm.SIZE2,), dtype=np.float64 )  # F, DX, DY, DX2, DXDY, DY2, DX3, DX2DY, DXDY2, DY3 at point xi
    for d in range(wlsqm.SIZE2):
        if knowns & (1 << d):
            fi[d] = funcs[d]( xi[0], xi[1] )  # fill in the known value  # TODO: add noise here too?

    # allocate array for sensitivity data
    #
    # for output; sens[k,j] = d(fi[j])/d(fk[k]) if f[i] unknown
    #                         nan if fi[j] known
    #
    # Note that if order=1, the part on second derivatives is not touched (so that an (nk,3) array
    # is valid); hence we pre-fill by nan.
    #
    if do_sens:
        sens = np.nan * np.empty( (nk,wlsqm.SIZE2), dtype=np.float64 )
    else:
        sens = None

    # fit the surrogate model (see wlsqm.fitter.simple for detailed documentation)
    #
    if debug:
        print  # blank line before debug info
    iterations_taken = wlsqm.fit_2D_iterative( xk, fk, xi, fi, sens, do_sens=do_sens, order=fit_order, knowns=knowns, debug=debug, weighting_method=weighting_method, max_iter=max_iter )
    print "refinement iterations taken: %d" % iterations_taken

    # check exact solution and relative error
    #
    exact = np.array( map( lambda func : func( xi[0], xi[1] ), funcs ) )
    err   = (fi - exact)

    print
    print "derivatives at xi:"
    print "exact:"
    print exact
    print "wlsqm solution:"
    print fi
    if do_sens:
        print "sensitivity:"
        print sens
    print "abs error:"
    print err
    print "rel error:"
    print (err / exact)

    #########################
    # plotting
    #########################

    xx   = np.linspace(axislims[0], axislims[1], nvis)
    yy   = np.linspace(axislims[2], axislims[3], nvis)
    X,Y  = np.meshgrid(xx, yy)
    W    = f(X,Y)

    # surrogate model - the returned fi[] are actually the coefficients of a polynomial
    model = wlsqm.lambdify_fit( xi, fi, dimension=2, order=fit_order )  # lambda x,y : ...
    xx2   = np.linspace(xi[0] - r, xi[0] + r, nvis)
    yy2   = np.linspace(xi[1] - r, xi[1] + r, nvis)
    X2,Y2 = np.meshgrid(xx2, yy2)
    W2    = model(X2,Y2)

#    # It is also possible to interpolate the model using the C API wrapper directly.
#    # The result is exactly the same; sometimes this API may be more convenient.
#    #
#    # Note that for the C API, the points x to which to interpolate the model must be formatted as x[k,:] = (xk,yk).
#    #
#    shp = np.shape(X2)
#    X2lin = np.reshape(X2, -1)
#    Y2lin = np.reshape(Y2, -1)
#    temp_x = np.array( [ (x,y) for x,y in zip(X2lin,Y2lin) ] )
#    out = wlsqm.interpolate_fit( xi, fi, dimension=2, order=fit_order, x=temp_x )
#    out = np.reshape( out, shp )
#    print
#    print "difference between Python and C API model interpolation:"
#    print out - W2  # should be close to zero

    print
    print "function values at neighbor points:"
    fxk = f( xk[:,0], xk[:,1] )
    mxk = model( xk[:,0], xk[:,1] )
    print "exact:"
    print fxk
    print "wlsqm solution:"
    print mxk
    print "abs error:"
    errf = mxk - fxk
    print errf
    print "rel error:"
    print (errf / fxk)

    # comparison
    W3   = f(X2,Y2)
    diff = W2 - W3  # fitted - exact
    idx  = np.argmax(np.abs( diff ))
    diff_lin = np.reshape(diff, -1)
    W3_lin   = np.reshape(W3, -1)
    maxerr_abs = diff_lin[idx]
    maxerr_rel = diff_lin[idx] / W3_lin[idx]
    print "largest absolute total fit error (over the domain of the fit, not just the neighbor points):"
    print "absolute: %g" % (maxerr_abs)
    print "relative: %g" % (maxerr_rel)

    fig = pl.figure(2, figsize=(12,6))  # for 2x1 subplots
#    fig = pl.figure(2, figsize=(12,12))
    fig.clf()

    ax = pl.subplot(1,2, 1)

    ax.plot( (xx[0],  xx[-1]), (yy[0],  yy[0]),  'k-' )
    ax.plot( (xx[-1], xx[-1]), (yy[0],  yy[-1]), 'k-' )
    ax.plot( (xx[0],  xx[-1]), (yy[-1], yy[-1]), 'k-' )
    ax.plot( (xx[0],  xx[0]),  (yy[0],  yy[-1]), 'k-' )

    ax.plot( xk[:,0], xk[:,1], linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )
    ax.plot( (xi[0] - r, xi[0] + r), (xi[1] - r, xi[1] - r), 'r-' )
    ax.plot( (xi[0] + r, xi[0] + r), (xi[1] - r, xi[1] + r), 'r-' )
    ax.plot( (xi[0] - r, xi[0] + r), (xi[1] + r, xi[1] + r), 'r-' )
    ax.plot( (xi[0] - r, xi[0] - r), (xi[1] - r, xi[1] + r), 'r-' )
    ax.plot( (xi[0],), (xi[1],), linestyle='none', marker='x', markeredgecolor='k', markerfacecolor='none' )
    pl.axis('tight')
    axis_marginize(ax, 0.02, 0.02)
    pl.grid(b=True, which='both')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.subplot(1,2, 2)

    # make 3d plot of the function
    #
    # see http://matplotlib.sourceforge.net/examples/mplot3d/lines3d_demo.html

    # Axes3D has a tendency to underestimate how much space it needs; it draws its labels
    # outside the window area in certain orientations.
    #
    # This causes the labels to be clipped, which looks bad. We prevent this by creating the axes
    # in a slightly smaller rect (leaving a margin). This way the labels will show - outside the Axes3D,
    # but still inside the figure window.
    #
    # The final touch is to set the window background to a matching white, so that the
    # background of the figure appears uniform.
    #
    fig.patch.set_color( (1,1,1) )
    fig.patch.set_alpha( 1.0 )
    x0y0wh = [ 0.02, 0.02, 0.96, 0.96 ]  # left, bottom, width, height      (here as fraction of subplot area)

    # compute the corresponding figure coordinates for the 2x1 subplot layout
    x0y0wh[0] = 0.5 + 0.5*x0y0wh[0]  # left
    x0y0wh[2] = 0.5*x0y0wh[2]        # width

    ax = p3.Axes3D(fig, rect=x0y0wh)

    stride = max(1, (nvis-1)//10)  # pick a good-looking stride (for lines; we actually have more vertices, making a smoother-looking curve between the lines)
    # use linewidth=0 to remove the wireframe if desired.
#    surf = ax.plot_surface(X,Y,W, rstride=stride, cstride=stride, cmap=matplotlib.cm.Blues_r, clim=[fmin,fmax], linewidth=0.25, alpha=0.5)
    ax.plot_wireframe(X,Y,W, rstride=stride, cstride=stride, color='k', linewidth=0.5, linestyle='solid')
#        pl.colorbar(surf, shrink=0.5, aspect=5)
#        pl.colorbar(surf, shrink=0.96)

    # sampled points
    if noise_eps > 0.:
        ax.plot( xk[:,0], xk[:,1], f( xk[:,0], xk[:,1] ), linestyle='none', marker='o', markeredgecolor='k', markerfacecolor='none' )  # exact
        ax.plot( xk[:,0], xk[:,1], fk[:], linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )  # including noise
    else:
        ax.plot( xk[:,0], xk[:,1], f( xk[:,0], xk[:,1] ), linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )  # exact

    # surrogate model
    ax.plot_wireframe(X2,Y2,W2, rstride=stride, cstride=stride, color='r', linewidth=0.5, linestyle='solid')

    # point xi
    ax.plot( (xi[0],), (xi[1],), f( xi[0], xi[1] ), linestyle='none', marker='x', markeredgecolor='k', markerfacecolor='none' )

#    ax.view_init(20, -48)
#    ax.view_init(18, -46)
#    ax.view_init(18, -128)
    ax.view_init(34, 140)

    ax.axis('tight')
    ax.set_zlim(-1.01, 1.01)
    pl.xlabel('$x$')
    pl.ylabel('$y$')
    ax.set_title('f(x,y)')


# one local model, 1D
#
def test1d():
    #########################
    # config
    #########################

    axislims = [0., 1., 0., 1.]  # [xmin, xmax, ymin, ymax], for plotting
    nvis = 101  # number of visualization points

    # Let's manufacture a solution (for which we know the derivatives analytically):
    #
#    expr     = sy.sympify("2*x")
#    expr     = sy.sympify("0.2*x")
#    expr     = sy.sympify("1.0 + 2*x + 4*x**2")
#    expr     = sy.sympify("0.1 + 0.2*x + 0.4*x**2")
    expr     = sy.sympify("sin(pi*x)")
#    expr     = sy.sympify("1 / (1 + x)")
#    expr     = sy.sympify("exp(x)")
#    expr     = sy.sympify("log(1 + x)")
#    expr     = sy.sympify("1.0 + 2*x + 4*x**2 + 7*x**3")

    noise_eps = 0#1e-3  # introduce this much Gaussian noise into each sampled function value (use 0. to turn off)

    xi = 0.45  # point x where we wish to find the derivatives

    # Degree of the surrogate polynomial.
    #
    # In the fit, when compared to the original function (if any is available),
    # usually the highest order will be nonsense, and the lower orders will be pretty accurate.
    #
    fit_order = 4  # 0 (constant), 1 (linear), 2 (quadratic), 3 (cubic) or 4 (quartic)

#    weighting_method = wlsqm.WEIGHT_UNIFORM  # best overall fit for function values
    weighting_method = wlsqm.WEIGHT_CENTER  # emphasize center to improve derivatives at the point xi

    max_iter = 100  # maximum number of refinement iterations for iterative fitting

    do_sens = False  # do sensitivity analysis of solution? ( d( fi[j] ) / d( fk[k] ) )

    debug = False#True  # print row scaling and condition number information? (if True, then do_sens must be False; the combination with both True is not supported)

    # Bitmask of what we know at point xi. In this example, just set the bits;
    # the data (from expr) will be automatically inserted into fi[].
    #
    # Bits from least sig. to most sig.: F, DX, DX2, DX3, DX4
    #
    knowns = 1

    # How many neighbor points to generate (to simulate the meshless 'grid').
    #
    # At least n_unknowns  points are needed to make the model fitting work at all
    # (but then the fit will be nonsensical, since it is possible to make the polynomial
    #  pass through exactly those points).
    #
    # n_unknows + 1  is the first value that makes the fitting overdetermined,
    # i.e. where the least-squares procedure starts providing any advantage.
    #
    # Here "unknown" means any element of fi[] not tagged as known in the "knowns" bitmask.
    #
    nk = 7  # used if grid_type == 'random'

    r  = 1e-1  # neighborhood radius

#    grid_type = 'random'
    grid_type = 'stencil'

    #########################
    # the test itself
    #########################

    print
    print "=" * 79
    print "1D case"
    print "=" * 79
    print

    print "expr: %s, xi = %s" % (expr, xi)
    labels = ["F", "DX", "DX2", "DX3", "DX4"]
    print "legend: %s" % ("\t".join(labels))
    knowns_str = ""
    for j in range(wlsqm.SIZE1):  # SIZE1 = maximum size of c matrix for 1D case
        if j > 0:
            knowns_str += '\t'
        if knowns & (1 << j):
            knowns_str += labels[j]
    print "knowns: %s" % knowns_str

    print "surrogate order: %d" % fit_order

    if noise_eps > 0.:
        print "simulating noisy input with eps = %g" % noise_eps

    # SymPy expr --> lambda(x)
    lambdify_numpy_1d = lambda expr: sy.lambdify(("x"), expr, modules="numpy")
    f       = lambdify_numpy_1d(expr)
    dfdx    = lambdify_numpy_1d(sy.diff(expr, "x"))
    d2fdx2  = lambdify_numpy_1d(sy.diff(expr, "x", 2))
    d3fdx3  = lambdify_numpy_1d(sy.diff(expr, "x", 3))
    d4fdx4  = lambdify_numpy_1d(sy.diff(expr, "x", 4))
    funcs = (f, dfdx, d2fdx2, d3fdx3, d4fdx4)  # list so we can refer to the functions by indices

    # create neighbor points xk around the point xi - this simulates our meshless 'grid'
    #
    if grid_type == 'random':
        xk = xi + r*2.*( np.random.sample( (nk,) ) - 0.5 )

    elif grid_type == 'stencil':
        points_per_axis = max(1,fit_order) + 1

        tt = np.linspace(-1., 1., points_per_axis)

        # reject the point at the origin if it is there
        point_list = [ x for x in tt if x != 0. ]

        xk = np.array( [ xi + r*p for p in point_list ] )
        nk = len(xk)

    else:
        raise ValueError("Unknown grid_type '%s'; valid: 'random', 'stencil'" % grid_type)


    # sample the function values at the neighbor points xk (these are used to fit the surrogate model)
    #
    sample_also_xi_str = " (and xi itself)" if knowns & 1 else ""
    print "sampling %d points%s" % (nk, sample_also_xi_str)
    fk = np.empty( (nk,), dtype=np.float64 )
    for k in xrange(nk):
        fk[k] = f( xk[k] )

    # simulate numerical errors by adding noise to the neighbor point function value samples
    #
    if noise_eps > 0.:
#        # uniform
#        noise = noise_eps*2.*(np.random.sample( np.shape(fk) ) - 0.5)

        # Gaussian, truncated
        mu    = 0.0
        sigma = noise_eps / 3.
        noise = np.random.normal( loc=mu, scale=sigma, size=np.shape(fk) )
        noise[noise < -3.*sigma] = -3.*sigma
        noise[noise > +3.*sigma] = +3.*sigma

        fk += noise

    # set knowns *at point xi*
    #
    # we use nan to spot unfilled entries
    fi = np.nan * np.empty( (wlsqm.SIZE1,), dtype=np.float64 )
    for d in range(wlsqm.SIZE1):
        if knowns & (1 << d):
            fi[d] = funcs[d]( xi )  # fill in the known value  # TODO: add noise here too?

    # allocate array for sensitivity data
    #
    # for output; sens[k,j] = d(fi[j])/d(fk[k]) if f[i] unknown
    #                         nan if fi[j] known
    #
    # Note that if order=1, the part on second derivatives is not touched (so that an (nk,3) array
    # is valid); hence we pre-fill by nan.
    #
    if do_sens:
        sens = np.nan * np.empty( (nk,wlsqm.SIZE1), dtype=np.float64 )
    else:
        sens = None

    # do the numerical differentiation
    #
    # xk : in, (nk,) array of neighbor point coordinates
    # fk : in, (nk,) array of function values at the neighbor points
    # xi : in, double, coordinate of the point xi
    # fi : in/out: if order=2, (3,) array containing (f, dfdx, d2fdx2) at point xi
    #              if order=1, (2,) array containing (f, dfdx) at point xi
    #      on input:  those elements must be filled that correspond to the bitmask "knowns".
    #      on output: the unknown elements will be filled in (leaving the knowns untouched).
    # sens : out: if order=2, (nk,3) array containing sensitivity information.
    #             if order=1, (nk,2) array containing sensitivity information.
    #             if fi[j] is unknown: sens[k,j] = d( fi[j] ) / d( fk[k] )
    #             if fi[j] is known:   sens[:,j] = nan (to indicate "not applicable").
    # order  : in, order of the surrogate polynomial. Can be 1 or 2.
    #          Linear fit gives first derivatives only and has O(h**2) error.
    #          Quadratic fit gives first and second derivatives and has O(h**3) error.
    # knowns : in, bitmask describing what is known about the function at the point xi.
    #          See the b1_* (bitmask, 1D case) constants.
    #
    if debug:
        print  # blank line before debug info
    iterations_taken = wlsqm.fit_1D_iterative( xk, fk, xi, fi, sens, do_sens=do_sens, order=fit_order, knowns=knowns, debug=debug, weighting_method=weighting_method, max_iter=max_iter )
    print "refinement iterations taken: %d" % iterations_taken

    # check exact solution and relative error
    #
    exact = np.array( map( lambda func : func( xi ), funcs ) )
    err   = (fi - exact)

    print
    print "derivatives at xi:"
    print "exact:"
    print exact
    print "wlsqm solution:"
    print fi
    if do_sens:
        print "sensitivity:"
        print sens
    print "abs error:"
    print err
    print "rel error:"
    print (err / exact)

    #########################
    # plotting
    #########################

    nvis = 10001
    xx   = np.linspace(axislims[0], axislims[1], nvis)
    ww   = f(xx)

    # surrogate model - the returned fi[] are actually the coefficients of a polynomial
    model = wlsqm.lambdify_fit( xi, fi, dimension=1, order=fit_order )  # lambda x : ...
    xx2   = np.linspace(xi - r, xi + r, nvis)
    ww2   = model(xx2)

#    # It is also possible to interpolate the model using the C API wrapper directly.
#    # The result is exactly the same; sometimes this API may be more convenient.
#    #
#    # Note that for the C API, the points x to which to interpolate the model must be formatted as x[:] = (xk).
#    #
#    out = wlsqm.interpolate_fit( xi, fi, dimension=1, order=fit_order, x=xx2 )
#    print
#    print "difference between Python and C API model interpolation:"
#    print out - ww2  # should be close to zero

    print
    print "function values (and derivatives) at neighbor points:"
    flags = [ wlsqm.i1_F, wlsqm.i1_X, wlsqm.i1_X2, wlsqm.i1_X3, wlsqm.i1_X4 ]
    for label,func,flag in zip(labels,funcs,flags):
        m   = wlsqm.lambdify_fit( xi, fi, dimension=1, order=fit_order, diff=flag )  # using diff=..., derivatives of the model can be lambdified, too
        fxk = func( xk )
        mxk = m( xk )
        print label
        print "exact:"
        print fxk
        print "wlsqm solution:"
        print mxk
        print "abs error:"
        errf = mxk - fxk
        print errf
        print "rel error:"
        print (errf / fxk)

    # comparison
    ww3  = f(xx2)
    diff = ww2 - ww3  # fitted - exact
    idx  = np.argmax(np.abs( diff ))
    maxerr_abs = diff[idx]
    maxerr_rel = diff[idx] / ww3[idx]
    print "largest absolute total fit error (over the domain of the fit, not just the neighbor points):"
    print "absolute: %g" % (maxerr_abs)
    print "relative: %g" % (maxerr_rel)

    fig = pl.figure(1, figsize=(6,6))
    fig.clf()

    ax = pl.subplot(1,1, 1)

    # the function
    ax.plot( xx, ww, color='k', linewidth=0.5, linestyle='solid' )

    # surrogate model
    ax.plot( xx2, ww2, color='r', linewidth=1., linestyle='solid' )

    # sampled points
    #
    fxk = f(xk)
    if noise_eps > 0.:
        ax.plot( xk, fxk, linestyle='none', marker='o', markeredgecolor='k', markerfacecolor='none' )  # exact
        ax.plot( xk, fk, linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )  # including noise
    else:
        ax.plot( xk, fxk, linestyle='none', marker='o', markeredgecolor='r', markerfacecolor='none' )  # exact

    # helper lines for easier reading of figure
#    ax.plot( np.tile(xk, (2,1)), np.tile((-0.05, 0.05), (nk,1)).T, 'k--' )
    tmp = np.zeros( (2,nk), dtype=np.float64 )  # for vertical lines from zero level to f(xk)
    tmp[1,:] = fxk
    ax.plot( np.tile(xk, (2,1)), tmp, 'r--', linewidth=0.25 )

    # sampled region
    ax.plot( (xi - r, xi + r), (0., 0.), 'r-', linewidth=2. )
    ax.plot( (xi - r, xi - r), (0., f(xi-r)), 'r--', linewidth=0.5 )
    ax.plot( (xi + r, xi + r), (0., f(xi+r)), 'r--', linewidth=0.5 )

    # point xi
    ax.plot( xi, f(xi), linestyle='none', marker='x', markeredgecolor='k', markerfacecolor='none' )

    pl.axis('tight')
    axis_marginize(ax, 0.02, 0.02)
    pl.grid(b=True, which='both')
    pl.xlabel('x')
    pl.ylabel('y')


def main():
    test3d()
    test2d()
    test1d()
    testmany2d()
#    wlsqm.test_pointer_wrappers()
    print
    pl.show()

if __name__ == '__main__':
    main()

