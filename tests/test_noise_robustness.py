"""Noise robustness of WLSQM first-derivative estimates.

A well-known property of WLSQM (and of weighted least squares in general):
the averaging effect of the fit denoises first derivatives significantly,
so moderate Gaussian noise on the function-value data still produces a
useful df/dx and df/dy estimate. The same is NOT true of second
derivatives — noise propagates into them at roughly 1/h**2, and for
sensible second derivatives one should run WLSQM once to get first
derivatives and then WLSQM again on those to get second derivatives.

That "fit once, differentiate twice" pattern is a usage strategy, not a
library feature; it lives in the randomthought project's README. This
file just pins the first-derivative claim: with ~1% noise on the function
values, the fitted first derivatives should still match truth to within a
tolerance well below the noise level.

Everything here is seeded (seed=42 as elsewhere) so failures reproduce.
"""

import numpy as np
import pytest

import wlsqm


def test_first_derivative_robust_to_gaussian_noise(rng):
    """2D linear function with 1% Gaussian noise on fk. The fit should
    recover df/dx and df/dy to well below the noise magnitude.

    Error bound: for N uncorrelated noisy samples, the least-squares
    first-derivative estimate has variance ~ sigma**2 / (N * var(x)). With
    N = 200, sigma = 0.01, and x-range ~ 1, the expected error on the slope
    is ~ 0.01 / sqrt(200 * 1/3) ~ 0.0012. A tolerance of 0.02 is generous
    and should be well above pathological seed-dependent runs while still
    failing if the fitter regresses by an order of magnitude.
    """
    def f_true(xy):
        return 2.0 * xy[..., 0] + 3.0 * xy[..., 1]
    npts = 200
    xk = rng.uniform(-1.0, 1.0, size=(npts, 2))
    sigma = 0.01
    noise = rng.normal(0.0, sigma, size=npts)
    fk = f_true(xk) + noise

    fi = np.zeros(wlsqm.number_of_dofs(2, 1))  # order 1 — only F, X, Y
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=1, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    # Truth: F = 0, X = 2, Y = 3
    tolerance = 0.02  # ~2 * the theoretical noise floor on a well-behaved seed
    assert abs(fi[wlsqm.i2_X] - 2.0) < tolerance, (
        f"df/dx off by {fi[wlsqm.i2_X] - 2.0:.4g} at sigma={sigma}"
    )
    assert abs(fi[wlsqm.i2_Y] - 3.0) < tolerance, (
        f"df/dy off by {fi[wlsqm.i2_Y] - 3.0:.4g} at sigma={sigma}"
    )
    # F at origin is also well-estimated (it's just the mean after detrending).
    assert abs(fi[wlsqm.i2_F]) < 0.02


def test_first_derivative_robust_on_quadratic_input(rng):
    """Same test, but the underlying function is quadratic and we fit at
    order 2. This is the realistic case: the WLSQM fit has to jointly
    denoise the gradient AND the Hessian, and we only care about the
    gradient. The gradient should still come out close to truth even
    though the Hessian does not.
    """
    def f_true(xy):
        x, y = xy[..., 0], xy[..., 1]
        return 1.0 + 2.0 * x + 3.0 * y + 0.5 * x ** 2 - y ** 2
    npts = 200
    xk = rng.uniform(-1.0, 1.0, size=(npts, 2))
    sigma = 0.01
    fk = f_true(xk) + rng.normal(0.0, sigma, size=npts)

    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    # First derivatives at origin: df/dx = 2, df/dy = 3.
    assert abs(fi[wlsqm.i2_X] - 2.0) < 0.05
    assert abs(fi[wlsqm.i2_Y] - 3.0) < 0.05
    # We deliberately DO NOT check the second derivatives here — that is the
    # whole point of this test. At sigma=0.01 with N=200, the Hessian
    # estimates can swing by order 0.1+ depending on the seed, and no
    # reasonable tolerance lets this test double as a Hessian regression.
