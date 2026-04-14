"""Edge cases: boundary orders (0, 4), minimum neighbor counts, knowns mask.

These tests exist mostly to guard against silent off-by-one errors or
silent no-op corner-case paths. If someone fits a 0th-order model (just the
function value at the origin), WLSQM should return the weighted average of
the neighbor function values, not garbage.
"""

import numpy as np
import pytest

import wlsqm


def test_fit_2d_order0_returns_weighted_average(rng):
    """Order 0 has only one DOF: the function value at xi. WLSQM with
    WEIGHT_UNIFORM and equal weights across all neighbors should return
    the arithmetic mean of fk."""
    xk = rng.uniform(-1.0, 1.0, size=(20, 2))
    fk = rng.standard_normal(20)
    fi = np.zeros(wlsqm.number_of_dofs(2, 0))
    assert fi.shape == (1,)
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=0, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    # Least-squares solution of the 1-unknown problem min_F sum_k (F - fk)^2
    # is simply the mean of fk.
    assert abs(fi[0] - fk.mean()) < 1e-12


def test_fit_2d_order4_recovers_quartic(rng):
    """Order 4 has 15 DOFs in 2D. Fit a pure quartic and verify the
    corresponding d4/dx4 value is recovered."""
    def f(xy):
        x, y = xy[..., 0], xy[..., 1]
        return x ** 4 + y ** 4  # both quartics, nothing else
    xk = rng.uniform(-1.0, 1.0, size=(40, 2))  # 40 >> 15 DOFs
    fk = f(xk)
    fi = np.zeros(wlsqm.number_of_dofs(2, 4))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=4, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    # d4f/dx4 = 24 for f = x^4; likewise d4f/dy4 = 24. Everything else 0.
    assert abs(fi[wlsqm.i2_X4] - 24.0) < 1e-8
    assert abs(fi[wlsqm.i2_Y4] - 24.0) < 1e-8
    # Lower-order terms at origin are all zero for f(x,y) = x^4 + y^4.
    assert abs(fi[wlsqm.i2_F])    < 1e-10
    assert abs(fi[wlsqm.i2_X])    < 1e-10
    assert abs(fi[wlsqm.i2_Y])    < 1e-10
    assert abs(fi[wlsqm.i2_X2])   < 1e-9
    assert abs(fi[wlsqm.i2_XY])   < 1e-9
    assert abs(fi[wlsqm.i2_Y2])   < 1e-9


def test_fit_2d_with_known_f_leaves_f_alone(rng):
    """When F is marked known, the fitter solves for the remaining DOFs
    while preserving fi[F] exactly as supplied."""
    def f(xy):
        x, y = xy[..., 0], xy[..., 1]
        return 1.0 + 2.0 * x + 3.0 * y  # linear
    xk = rng.uniform(-1.0, 1.0, size=(15, 2))
    fk = f(xk)
    fi = np.zeros(wlsqm.number_of_dofs(2, 1))
    fi[wlsqm.i2_F] = 999.0  # deliberately wrong — we claim this IS the truth
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=1, knowns=wlsqm.b2_F,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert fi[wlsqm.i2_F] == 999.0  # untouched
    # The gradient fit should still recover approximately (2, 3), but because
    # we pinned the constant wrong, there's a large residual. We only check
    # that fi[F] was not overwritten.


def test_fit_1d_minimum_neighbor_count_order2():
    """Order 2 in 1D has 3 DOFs. A fit on exactly 3 points is determined.
    Use a symmetric stencil {-h, 0, h} for a trivial verification."""
    h = 0.1
    xk = np.array([-h, 0.0, h])
    fk = np.array([1.0, 0.5, 2.0])  # arbitrary
    fi = np.zeros(wlsqm.number_of_dofs(1, 2))
    wlsqm.fit_1D(
        xk=xk, fk=fk, xi=0.0, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    # Classical central differences:
    assert abs(fi[wlsqm.i1_F]  - 0.5)                  < 1e-12
    assert abs(fi[wlsqm.i1_X]  - (2.0 - 1.0) / (2 * h))  < 1e-12
    assert abs(fi[wlsqm.i1_X2] - (1.0 + 2.0 - 2 * 0.5) / (h * h)) < 1e-10
