"""Shared fixtures and helpers for the wlsqm test suite.

The fixtures here intentionally stay small — most tests want a seeded random
number generator and a handful of analytical polynomial evaluators. Anything
more specific lives next to the test that needs it.

Seed is fixed at 42 (the standard across this project family) so that failures
reproduce identically across runs and CI.
"""

from __future__ import annotations

import numpy as np
import pytest


SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """Fresh seeded NumPy Generator, one per test."""
    return np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Analytical polynomial helpers
# ---------------------------------------------------------------------------
#
# The convention throughout this suite: every `poly_*` helper returns `(f, fi)`
# where `f(x_or_xy_or_xyz)` evaluates the polynomial and `fi` is the
# coefficient array that a successful WLSQM fit should recover when the fit
# origin is (0, 0, 0) and the fit order matches the polynomial order.
#
# Important: `fi` uses wlsqm's "partially baked" convention — the entries hold
# DERIVATIVE VALUES at the origin, not monomial coefficients. Hence the 2! and
# 6 and 24 factors versus the raw polynomial.


def poly2d_order2():
    """f(x, y) = 1 + 2x + 3y + 4xy + 5x^2 + 6y^2 evaluated at the origin."""
    def f(xy):
        x, y = xy[..., 0], xy[..., 1]
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * y + 5.0 * x ** 2 + 6.0 * y ** 2
    # DOF ordering at order=2: F, X, Y, X2, XY, Y2
    #   F   = f(0,0)     = 1
    #   X   = df/dx      = 2
    #   Y   = df/dy      = 3
    #   X2  = d2f/dx2    = 2 * 5 = 10   (polynomial coef 5, derivative 10)
    #   XY  = d2f/dxdy   = 4
    #   Y2  = d2f/dy2    = 2 * 6 = 12
    fi = np.array([1.0, 2.0, 3.0, 10.0, 4.0, 12.0])
    return f, fi


def poly3d_order2():
    """f(x,y,z) = 1 + 2x - y + 3z + x*y at the origin."""
    def f(xyz):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        return 1.0 + 2.0 * x - y + 3.0 * z + x * y
    # DOF ordering at order=2 (3D): F, X, Y, Z, X2, XY, Y2, YZ, Z2, XZ
    fi = np.array([1.0, 2.0, -1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    return f, fi


def poly1d_order2():
    """f(x) = 1 + 2x + 3x^2 at the origin; d2f/dx2 = 6."""
    def f(x):
        return 1.0 + 2.0 * x + 3.0 * x ** 2
    # DOF ordering at order=2 (1D): F, X, X2
    fi = np.array([1.0, 2.0, 6.0])
    return f, fi


def poly2d_order3():
    """f(x,y) = 1 + x - 2y + 3x^2 - xy + 2y^2 + x^3 - 4x^2 y + y^3 at origin."""
    def f(xy):
        x, y = xy[..., 0], xy[..., 1]
        return (1.0 + x - 2.0 * y
                + 3.0 * x ** 2 - x * y + 2.0 * y ** 2
                + x ** 3 - 4.0 * x ** 2 * y + y ** 3)
    # DOF ordering at order=3 (2D): F, X, Y, X2, XY, Y2, X3, X2Y, XY2, Y3
    #   F   = 1
    #   X   = 1
    #   Y   = -2
    #   X2  = 2*3        = 6
    #   XY  = -1
    #   Y2  = 2*2        = 4
    #   X3  = 6*1        = 6      (d3f/dx3 of x^3 is 6)
    #   X2Y = 2*(-4)     = -8     (d3f/dx2 dy of x^2 y is 2, coefficient -4)
    #   XY2 = 0                   (no xy^2 term in polynomial)
    #   Y3  = 6*1        = 6
    fi = np.array([1.0, 1.0, -2.0, 6.0, -1.0, 4.0, 6.0, -8.0, 0.0, 6.0])
    return f, fi
