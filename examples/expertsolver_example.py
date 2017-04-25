# -*- coding: utf-8 -*-
"""A minimal usage example for ExpertSolver.

JJ 2017-03-28
"""

from __future__ import division, print_function, absolute_import

import numpy as np

import scipy.spatial.ckdtree

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

import wlsqm


def project_onto_regular_grid_2D(x, F, nvis=101, fit_order=1, nk=10):
    """Project scalar data from a 2D point cloud onto a regular grid.

Useful for plotting. Uses the WLSQM meshless method.

The bounding box of the x data is automatically used as the bounds of the generated regular grid.

Parameters:
    x : rank-2 array, dtype np.float64
        Point cloud, one point per row. x[i,:] = (xi,yi)

    F : rank-1 array, dtype np.float64
        The corresponding function values. F[i] = F( x[i,:] )

    nvis : int
        Number of points per axis in the generated regular grid.

    fit_order : int
        Order of the surrogate polynomial, one of [0,1,2,3,4].

    nk : int
        Number of nearest neighbors to use for fitting the model.

Return value:
    tuple (X,Y,Z)
        where
        X,Y are rank-2 meshgrid arrays representing the generated regular grid, and
        Z is an array of the same shape, containing the corresponding data values.

"""
    # Form the neighborhoods.

    # index the input points for fast searching
    tree = scipy.spatial.cKDTree( data=x )

    # Find max_nk nearest neighbors of each input point.
    #
    # The +1 is for the point itself, since it is always the nearest to itself.
    #
    # (cKDTree.query() supports querying for arbitrary x; here we just set these x as the same as the points in the tree.)
    #
    dd,ii = tree.query( x, 1 + nk )

    # Take only the neighbors of points[i], excluding the point itself.
    #
    ii = ii[:,1:]  # points[ ii[i,k] ] is the kth nearest neighbor of points[i]. Shape of ii is (npoints, nk).

    # neighbor point indices (pointing to rows in x[]); typecast to int32
    hoods = np.array( ii, dtype=np.int32 )

    npoints  = x.shape[0]
    nk_array = nk * np.ones( (npoints,), dtype=np.int32 )  # number of neighbors, i.e. nk_array[i] is the number of actually used columns in hoods[i,:]

    # Construct the model by least-squares fitting
    #
    fit_order_array = fit_order            * np.ones( (npoints,), dtype=np.int32 )
    knowns_array    = wlsqm.b2_F           * np.ones( (npoints,), dtype=np.int64 )  # bitmask! wlsqm.b*
    wm_array        = wlsqm.WEIGHT_UNIFORM * np.ones( (npoints,), dtype=np.int32 )
    solver = wlsqm.ExpertSolver( dimension=2,
                                 nk=nk_array,
                                 order=fit_order_array,
                                 knowns=knowns_array,
                                 weighting_method=wm_array,
                                 algorithm=wlsqm.ALGO_BASIC,
                                 do_sens=False,
                                 max_iter=10,  # must be an int even though this parameter is not used in ALGO_BASIC mode
                                 ntasks=8,
                                 debug=False )

    no = wlsqm.number_of_dofs( dimension=2, order=fit_order )
    fi = np.empty( (npoints,no), dtype=np.float64 )
    fi[:,0] = F  # fi[i,0] contains the function value at point x[i,:]

    solver.prepare( xi=x, xk=x[hoods] )  # generate problem matrices from the geometry of the point cloud
    solver.solve( fk=fi[hoods,0], fi=fi, sens=None )  # compute least-squares fit to data


    # generate the regular grid for output
    #
    xx  = np.linspace( np.min(x[:,0]), np.max(x[:,0]), nvis )
    yy  = np.linspace( np.min(x[:,1]), np.max(x[:,1]), nvis )
    X,Y = np.meshgrid(xx,yy)

    # make a flat list of grid points (rank-2 array, one point per row)
    #
    Xlin = np.reshape(X, -1)
    Ylin = np.reshape(Y, -1)
    xout = np.empty( (len(Xlin), 2), dtype=np.float64 )
    xout[:,0] = Xlin
    xout[:,1] = Ylin

    # Using the model, interpolate onto the regular grid
    #
    solver.prep_interpolate()  # prepare global model
    Z,mi = solver.interpolate( xout, mode='nearest' )  # use the nearest local model; fast, surprisingly accurate
                                                       # if a reasonable number of points (and continuous-looking
                                                       # although technically has jumps over Voronoi cell boundaries)
    # when mode="nearest", "mi" is an array containing the index of the local model (which belongs to x[mi,:]) used for each evaluation

    return (X, Y, np.reshape( Z, X.shape ))


def plot_wireframe( data, figno=None ):
    """Make and label a wireframe plot.

Parameters:
    data : dict
        key   : "x","y","z"
        value : tuple (rank-2 array in meshgrid format, axis label)

Return value:
    ax
        The Axes3D object that was used for plotting.
"""
    # http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure(figno)

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

    ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig, rect=x0y0wh)

    X,xlabel = data["x"]
    Y,ylabel = data["y"]
    Z,zlabel = data["z"]
    ax.plot_wireframe( X, Y, Z )

    ax.view_init(34, -40)
    ax.axis('tight')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(zlabel)

    return ax


def main():
    x = np.random.random( (1000, 2) )                # point cloud (no mesh topology!)
    F = np.sin(np.pi*x[:,0]) * np.cos(np.pi*x[:,1])  # function values on the point cloud
    X,Y,Z = project_onto_regular_grid_2D(x, F, fit_order=2, nk=30)
    plot_wireframe( {"x" : (X, r"$x$"),
                     "y" : (Y, r"$y$"),
                     "z" : (Z, r"$f(x,y)$")} )

if __name__ == '__main__':
    main()
    plt.show()
