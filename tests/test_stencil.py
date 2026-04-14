"""Finite-difference stencil equivalence.

A WLSQM fit on a classical finite-difference stencil should reproduce the
exact stencil result for ANY smooth input, not just polynomials. The fit
over a stencil is a determined square linear system in the monomial basis,
and the classical stencil coefficients are literally the rows of its inverse
— so WLSQM's answer has to match the hand-coded stencil formula to roundoff.

This file tests the three classical central-difference stencils on
non-polynomial inputs (trigonometric + exponential). If these ever break,
the WLSQM fit has drifted away from implementing the stencil it was
mathematically designed to generalize.

Stencils used:
  1D: 3-point central difference (center, ±h)
  2D: 5-point plus-shaped (center, ±h along x, ±h along y; XY term pinned
      to zero via the `knowns` bitmask — the plus stencil carries no mixed
      derivative information)
  3D: 7-point plus-shaped (center, ±h along x/y/z; XY, YZ, XZ pinned to 0)
"""

import numpy as np
import pytest

import wlsqm


# Stencil half-width. Both WLSQM and the hand-coded central-difference
# stencil get the same O(h**2) truncation error (which cancels when we compare
# them to each other), but second-derivative estimates suffer floating-point
# cancellation of order eps / h**2. With h = 1e-2 that's ~1e-12, well inside
# our tolerance budget. Do NOT shrink h below ~1e-3 without loosening the
# second-derivative tolerances.
H = 1e-2


# ---------------------------------------------------------------------------
# 1D: 3-point central difference
# ---------------------------------------------------------------------------

def _central_diff_1d(f, x0, h):
    """Classical 3-point central differences at x0: (F, df/dx, d2f/dx2)."""
    f_m, f_c, f_p = f(x0 - h), f(x0), f(x0 + h)
    F = f_c
    dX = (f_p - f_m) / (2.0 * h)
    dX2 = (f_p - 2.0 * f_c + f_m) / (h * h)
    return F, dX, dX2


@pytest.mark.parametrize(
    "f,x0",
    [
        (lambda x: np.sin(x),         0.3),
        (lambda x: np.exp(x),        -0.2),
        (lambda x: np.sin(x) * np.exp(x), 0.5),
    ],
)
def test_stencil_1d_central_difference(f, x0):
    # Vectorize since f is defined for scalar input.
    f_vec = np.vectorize(f, otypes=[np.float64])

    # Stencil values (by hand).
    F_ref, dX_ref, dX2_ref = _central_diff_1d(f, x0, H)

    # WLSQM fit on the same three stencil points — order 2, 3 DOFs, 3 points
    # -> determined system.
    xk = np.array([x0 - H, x0, x0 + H], dtype=np.float64)
    fk = f_vec(xk)
    fi = np.zeros(wlsqm.number_of_dofs(1, 2))
    wlsqm.fit_1D(
        xk=xk, fk=fk, xi=x0, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    assert abs(fi[wlsqm.i1_F]  - F_ref)   < 1e-12
    assert abs(fi[wlsqm.i1_X]  - dX_ref)  < 1e-11
    assert abs(fi[wlsqm.i1_X2] - dX2_ref) < 1e-10


# ---------------------------------------------------------------------------
# 2D: 5-point plus-shaped stencil
# ---------------------------------------------------------------------------

def _plus_stencil_2d(f, x0, y0, h):
    """5-point plus stencil at (x0, y0). Returns (F, dx, dy, dxx, dyy).

    The plus stencil carries no mixed-derivative information, so dxy is
    undefined and we pin it to 0 in the WLSQM fit via the `knowns` bitmask.
    """
    f_c  = f(x0, y0)
    f_xp = f(x0 + h, y0)
    f_xm = f(x0 - h, y0)
    f_yp = f(x0, y0 + h)
    f_ym = f(x0, y0 - h)
    F   = f_c
    dX  = (f_xp - f_xm) / (2.0 * h)
    dY  = (f_yp - f_ym) / (2.0 * h)
    dX2 = (f_xp - 2.0 * f_c + f_xm) / (h * h)
    dY2 = (f_yp - 2.0 * f_c + f_ym) / (h * h)
    return F, dX, dY, dX2, dY2


@pytest.mark.parametrize(
    "f,x0,y0",
    [
        (lambda x, y: np.sin(x) + np.cos(y),              0.3,  0.4),
        (lambda x, y: np.exp(-0.5 * (x * x + y * y)),     0.1, -0.2),
        (lambda x, y: np.sin(x * y),                      0.3,  0.2),
    ],
)
def test_stencil_2d_plus_shape(f, x0, y0):
    F_ref, dX_ref, dY_ref, dX2_ref, dY2_ref = _plus_stencil_2d(f, x0, y0, H)

    xk = np.array(
        [[x0,     y0    ],
         [x0 + H, y0    ],
         [x0 - H, y0    ],
         [x0,     y0 + H],
         [x0,     y0 - H]],
        dtype=np.float64,
    )
    fk = np.array([f(px, py) for px, py in xk], dtype=np.float64)

    # 2D order 2: 6 DOFs (F, X, Y, X2, XY, Y2). We pin XY = 0 as a known,
    # leaving 5 unknowns — matching the 5 stencil points.
    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    fi[wlsqm.i2_XY] = 0.0  # pinned value
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([x0, y0]), fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=wlsqm.b2_XY,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    assert abs(fi[wlsqm.i2_F]  - F_ref)   < 1e-10
    assert abs(fi[wlsqm.i2_X]  - dX_ref)  < 1e-10
    assert abs(fi[wlsqm.i2_Y]  - dY_ref)  < 1e-10
    assert abs(fi[wlsqm.i2_X2] - dX2_ref) < 1e-8
    assert abs(fi[wlsqm.i2_Y2] - dY2_ref) < 1e-8
    # XY was marked known; must still be 0.
    assert fi[wlsqm.i2_XY] == 0.0


# ---------------------------------------------------------------------------
# 3D: 7-point plus-shaped stencil
# ---------------------------------------------------------------------------

def _plus_stencil_3d(f, x0, y0, z0, h):
    f_c  = f(x0, y0, z0)
    f_xp = f(x0 + h, y0, z0); f_xm = f(x0 - h, y0, z0)
    f_yp = f(x0, y0 + h, z0); f_ym = f(x0, y0 - h, z0)
    f_zp = f(x0, y0, z0 + h); f_zm = f(x0, y0, z0 - h)
    F   = f_c
    dX  = (f_xp - f_xm) / (2.0 * h)
    dY  = (f_yp - f_ym) / (2.0 * h)
    dZ  = (f_zp - f_zm) / (2.0 * h)
    dX2 = (f_xp - 2.0 * f_c + f_xm) / (h * h)
    dY2 = (f_yp - 2.0 * f_c + f_ym) / (h * h)
    dZ2 = (f_zp - 2.0 * f_c + f_zm) / (h * h)
    return F, dX, dY, dZ, dX2, dY2, dZ2


@pytest.mark.parametrize(
    "f,x0,y0,z0",
    [
        (lambda x, y, z: np.sin(x) * np.cos(y) * np.exp(z), 0.2, 0.3, -0.1),
        (lambda x, y, z: np.exp(-0.5 * (x*x + y*y + z*z)), 0.1, -0.2, 0.3),
    ],
)
def test_stencil_3d_plus_shape(f, x0, y0, z0):
    F_ref, dX_ref, dY_ref, dZ_ref, dX2_ref, dY2_ref, dZ2_ref = _plus_stencil_3d(
        f, x0, y0, z0, H,
    )

    xk = np.array(
        [[x0,     y0,     z0    ],
         [x0 + H, y0,     z0    ],
         [x0 - H, y0,     z0    ],
         [x0,     y0 + H, z0    ],
         [x0,     y0 - H, z0    ],
         [x0,     y0,     z0 + H],
         [x0,     y0,     z0 - H]],
        dtype=np.float64,
    )
    fk = np.array([f(px, py, pz) for px, py, pz in xk], dtype=np.float64)

    # 3D order 2: 10 DOFs. The plus stencil supplies no mixed-derivative info,
    # so pin XY, YZ, XZ to zero — leaving 7 unknowns matching 7 stencil points.
    fi = np.zeros(wlsqm.number_of_dofs(3, 2))
    fi[wlsqm.i3_XY] = 0.0
    fi[wlsqm.i3_YZ] = 0.0
    fi[wlsqm.i3_XZ] = 0.0
    wlsqm.fit_3D(
        xk=xk, fk=fk, xi=np.array([x0, y0, z0]), fi=fi,
        sens=None, do_sens=False,
        order=2,
        knowns=(wlsqm.b3_XY | wlsqm.b3_YZ | wlsqm.b3_XZ),
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    assert abs(fi[wlsqm.i3_F]  - F_ref)   < 1e-10
    assert abs(fi[wlsqm.i3_X]  - dX_ref)  < 1e-10
    assert abs(fi[wlsqm.i3_Y]  - dY_ref)  < 1e-10
    assert abs(fi[wlsqm.i3_Z]  - dZ_ref)  < 1e-10
    assert abs(fi[wlsqm.i3_X2] - dX2_ref) < 1e-6
    assert abs(fi[wlsqm.i3_Y2] - dY2_ref) < 1e-6
    assert abs(fi[wlsqm.i3_Z2] - dZ2_ref) < 1e-6
