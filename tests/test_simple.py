"""Simple-API (wlsqm.fit_?D) polynomial-recovery tests.

For each dimension and order we fit a known polynomial on a scattered point
cloud. Since the polynomial is exactly representable in the monomial basis
used by WLSQM, the fit should recover every derivative at the origin to
well inside machine precision.
"""

import numpy as np

import wlsqm

from .conftest import (  # noqa: E402
    poly1d_order2,
    poly2d_order2,
    poly2d_order3,
    poly3d_order2,
)


# Every exact-polynomial recovery should be at least this tight. The actual
# numbers we see in practice are O(1e-13) or smaller for order <= 3 on a
# well-conditioned random cloud; 1e-10 gives comfortable headroom.
ATOL_EXACT = 1e-10


def _cloud_2d(rng, npts=30, radius=1.0):
    return rng.uniform(-radius, radius, size=(npts, 2))


def _cloud_3d(rng, npts=40, radius=1.0):
    return rng.uniform(-radius, radius, size=(npts, 3))


def _cloud_1d(rng, npts=15, radius=1.0):
    return rng.uniform(-radius, radius, size=(npts,))


def test_fit_1d_order2_recovers_derivatives(rng):
    f, fi_expected = poly1d_order2()
    xk = _cloud_1d(rng)
    fk = f(xk)

    fi = np.zeros(wlsqm.number_of_dofs(1, 2))
    wlsqm.fit_1D(
        xk=xk, fk=fk, xi=0.0, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert np.allclose(fi, fi_expected, atol=ATOL_EXACT), (
        f"got {fi}, expected {fi_expected}"
    )


def test_fit_2d_order2_recovers_derivatives(rng):
    f, fi_expected = poly2d_order2()
    xk = _cloud_2d(rng)
    fk = f(xk)

    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert np.allclose(fi, fi_expected, atol=ATOL_EXACT), (
        f"got {fi}, expected {fi_expected}"
    )


def test_fit_3d_order2_recovers_derivatives(rng):
    f, fi_expected = poly3d_order2()
    xk = _cloud_3d(rng)
    fk = f(xk)

    fi = np.zeros(wlsqm.number_of_dofs(3, 2))
    wlsqm.fit_3D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert np.allclose(fi, fi_expected, atol=ATOL_EXACT), (
        f"got {fi}, expected {fi_expected}"
    )


def test_fit_2d_order3_recovers_derivatives(rng):
    """Order-3 2D polynomial recovery. Exercises the order=3 matrix layout."""
    f, fi_expected = poly2d_order3()
    # Need more neighbor points at order=3 (10 DOFs instead of 6).
    xk = _cloud_2d(rng, npts=50)
    fk = f(xk)

    fi = np.zeros(wlsqm.number_of_dofs(2, 3))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=3, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert np.allclose(fi, fi_expected, atol=ATOL_EXACT), (
        f"got {fi}, expected {fi_expected}"
    )


def test_fit_2d_weight_center_also_recovers(rng):
    """WEIGHT_CENTER emphasizes points near the origin of the fit, but for
    an exact polynomial fit the choice of weighting cannot affect the result
    (the fit is exact either way)."""
    f, fi_expected = poly2d_order2()
    xk = _cloud_2d(rng)
    fk = f(xk)

    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=np.array([0.0, 0.0]), fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_CENTER,
        debug=False,
    )
    assert np.allclose(fi, fi_expected, atol=ATOL_EXACT)


def test_fit_2d_many_same_as_loop(rng):
    """Batched fit_2D_many produces the same result as a per-case loop."""
    f, fi_expected = poly2d_order2()
    ncases = 8
    npts = 25
    xk_all = rng.uniform(-1.0, 1.0, size=(ncases, npts, 2))
    fk_all = np.stack([f(xk_all[j]) for j in range(ncases)])

    # Single-case reference via fit_2D in a Python loop.
    fi_ref = np.zeros((ncases, wlsqm.number_of_dofs(2, 2)))
    for j in range(ncases):
        wlsqm.fit_2D(
            xk=xk_all[j], fk=fk_all[j], xi=np.array([0.0, 0.0]), fi=fi_ref[j],
            sens=None, do_sens=False,
            order=2, knowns=0,
            weighting_method=wlsqm.WEIGHT_UNIFORM,
            debug=False,
        )

    # Batched fit.
    fi_many = np.zeros((ncases, wlsqm.number_of_dofs(2, 2)))
    nk = np.full(ncases, npts, dtype=np.int32)
    order = np.full(ncases, 2, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)
    xi = np.zeros((ncases, 2))
    wlsqm.fit_2D_many(
        xk=xk_all, fk=fk_all, nk=nk, xi=xi, fi=fi_many,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        debug=False,
    )

    assert np.allclose(fi_many, fi_ref, atol=1e-14)
    # And each row is the expected polynomial derivatives.
    for j in range(ncases):
        assert np.allclose(fi_many[j], fi_expected, atol=ATOL_EXACT)
