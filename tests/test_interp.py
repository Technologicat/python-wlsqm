"""Interpolation tests.

Given a fitted local polynomial model, `wlsqm.interpolate_fit` evaluates
it (or any of its derivatives) at arbitrary query points. For an exact
polynomial fit, the interpolated values must equal the analytical values
to machine precision everywhere inside the domain of the fit — not just
at the fit origin.
"""

import numpy as np

import wlsqm

from conftest import poly1d_order2, poly2d_order2, poly3d_order2


def test_interp_2d_function_value_matches_analytical(rng):
    f, _ = poly2d_order2()
    # Fit
    xk = rng.uniform(-1.0, 1.0, size=(30, 2))
    fk = f(xk)
    xi = np.array([0.0, 0.0])
    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    # Evaluate at fresh query points inside the neighborhood.
    query = rng.uniform(-0.8, 0.8, size=(20, 2))
    values = wlsqm.interpolate_fit(xi, fi, dimension=2, order=2, x=query, diff=wlsqm.i2_F)
    expected = f(query)
    assert np.allclose(values, expected, atol=1e-10)


def test_interp_2d_partial_derivatives(rng):
    f, fi_expected = poly2d_order2()
    xk = rng.uniform(-1.0, 1.0, size=(30, 2))
    fk = f(xk)
    xi = np.array([0.0, 0.0])
    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    # Analytical derivatives of f(x,y) = 1 + 2x + 3y + 4xy + 5x^2 + 6y^2:
    #   df/dx     = 2 + 4y + 10x
    #   df/dy     = 3 + 4x + 12y
    #   d2f/dx2   = 10
    #   d2f/dxdy  = 4
    #   d2f/dy2   = 12
    query = rng.uniform(-0.8, 0.8, size=(15, 2))
    qx, qy = query[:, 0], query[:, 1]

    got_dx  = wlsqm.interpolate_fit(xi, fi, 2, 2, query, diff=wlsqm.i2_X)
    got_dy  = wlsqm.interpolate_fit(xi, fi, 2, 2, query, diff=wlsqm.i2_Y)
    got_dxx = wlsqm.interpolate_fit(xi, fi, 2, 2, query, diff=wlsqm.i2_X2)
    got_dxy = wlsqm.interpolate_fit(xi, fi, 2, 2, query, diff=wlsqm.i2_XY)
    got_dyy = wlsqm.interpolate_fit(xi, fi, 2, 2, query, diff=wlsqm.i2_Y2)

    assert np.allclose(got_dx,  2.0 + 4.0 * qy + 10.0 * qx, atol=1e-10)
    assert np.allclose(got_dy,  3.0 + 4.0 * qx + 12.0 * qy, atol=1e-10)
    assert np.allclose(got_dxx, np.full_like(qx, 10.0),     atol=1e-10)
    assert np.allclose(got_dxy, np.full_like(qx,  4.0),     atol=1e-10)
    assert np.allclose(got_dyy, np.full_like(qx, 12.0),     atol=1e-10)


def test_interp_3d_function_value(rng):
    f, _ = poly3d_order2()
    xk = rng.uniform(-1.0, 1.0, size=(40, 3))
    fk = f(xk)
    xi = np.array([0.0, 0.0, 0.0])
    fi = np.zeros(wlsqm.number_of_dofs(3, 2))
    wlsqm.fit_3D(
        xk=xk, fk=fk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    query = rng.uniform(-0.8, 0.8, size=(20, 3))
    values = wlsqm.interpolate_fit(xi, fi, dimension=3, order=2, x=query, diff=wlsqm.i3_F)
    assert np.allclose(values, f(query), atol=1e-10)


def test_interp_1d_matches_analytical(rng):
    f, _ = poly1d_order2()
    xk = rng.uniform(-1.0, 1.0, size=(15,))
    fk = f(xk)
    fi = np.zeros(wlsqm.number_of_dofs(1, 2))
    wlsqm.fit_1D(
        xk=xk, fk=fk, xi=0.0, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    query = rng.uniform(-0.8, 0.8, size=(10,))
    values = wlsqm.interpolate_fit(0.0, fi, dimension=1, order=2, x=query, diff=wlsqm.i1_F)
    assert np.allclose(values, f(query), atol=1e-12)
