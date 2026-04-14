"""Parallel-vs-serial equivalence + regression test for the 1D parallel bug.

The `*_many_parallel` fitting routines run independent problem instances in
an OpenMP `prange` with per-thread work buffers. They must produce bit-for-bit
the same fi arrays as the corresponding serial loops. The 1D branch of
`generic_fit_basic_many_parallel` in particular had a typo that used the
module-level `TASKID` (compile-time 0) instead of `taskid = omp_get_thread_num()`,
so every thread clobbered work buffer 0 and the fits came out silently wrong
whenever ntasks > 1 — fixed during the DEF removal pass. This file guards
that regression.
"""

import numpy as np
import pytest

import wlsqm


def _make_2d_batch(rng, ncases, npts):
    xk = rng.uniform(-1.0, 1.0, size=(ncases, npts, 2))
    # f_j(x, y) = j + (j + 1) * x - 2 * y — each case a different affine plane
    fk = np.stack([
        j + (j + 1) * xk[j, :, 0] - 2.0 * xk[j, :, 1]
        for j in range(ncases)
    ])
    return xk, fk


def _make_1d_batch(rng, ncases, npts):
    xk = rng.uniform(-1.0, 1.0, size=(ncases, npts))
    # f_j(x) = j + (j + 1) * x — per-case line
    fk = np.stack([j + (j + 1) * xk[j] for j in range(ncases)])
    return xk, fk


def test_2d_parallel_matches_serial(rng):
    ncases, npts = 32, 25
    xk, fk = _make_2d_batch(rng, ncases, npts)

    xi = np.zeros((ncases, 2))
    nk = np.full(ncases, npts, dtype=np.int32)
    order = np.ones(ncases, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)

    fi_serial = np.zeros((ncases, wlsqm.number_of_dofs(2, 1)))
    wlsqm.fit_2D_many(
        xk=xk, fk=fk, nk=nk, xi=xi, fi=fi_serial,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        debug=False,
    )

    fi_parallel = np.zeros_like(fi_serial)
    wlsqm.fit_2D_many_parallel(
        xk=xk, fk=fk, nk=nk, xi=xi, fi=fi_parallel,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        ntasks=4, debug=False,
    )

    assert np.allclose(fi_parallel, fi_serial, atol=1e-14)
    # And every case recovered its analytical derivatives: F=j, X=j+1, Y=-2.
    for j in range(ncases):
        assert abs(fi_parallel[j, 0] - j) < 1e-10
        assert abs(fi_parallel[j, 1] - (j + 1)) < 1e-10
        assert abs(fi_parallel[j, 2] - (-2.0)) < 1e-10


def test_1d_parallel_many_case_regression(rng):
    """Guard against the pre-existing data-race bug where the 1D parallel
    basic branch used TASKID=0 instead of the per-thread `taskid`. With the
    bug, every OpenMP thread wrote to work buffer 0, corrupting fits for all
    but (accidentally) one case per timing window. 64 cases × 4 threads is
    plenty of pressure for the bug to surface if it regresses.
    """
    ncases, npts = 64, 25
    xk, fk = _make_1d_batch(rng, ncases, npts)

    xi = np.zeros(ncases)
    nk = np.full(ncases, npts, dtype=np.int32)
    order = np.ones(ncases, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)

    fi = np.zeros((ncases, wlsqm.number_of_dofs(1, 1)))
    wlsqm.fit_1D_many_parallel(
        xk=xk, fk=fk, nk=nk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        ntasks=4, debug=False,
    )

    for j in range(ncases):
        assert abs(fi[j, 0] - j) < 1e-10, f"case {j}: F got {fi[j, 0]}"
        assert abs(fi[j, 1] - (j + 1)) < 1e-10, f"case {j}: X got {fi[j, 1]}"


def test_1d_parallel_matches_serial(rng):
    ncases, npts = 20, 30
    xk, fk = _make_1d_batch(rng, ncases, npts)

    xi = np.zeros(ncases)
    nk = np.full(ncases, npts, dtype=np.int32)
    order = np.full(ncases, 2, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)

    fi_serial = np.zeros((ncases, wlsqm.number_of_dofs(1, 2)))
    wlsqm.fit_1D_many(
        xk=xk, fk=fk, nk=nk, xi=xi, fi=fi_serial,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        debug=False,
    )

    fi_parallel = np.zeros_like(fi_serial)
    wlsqm.fit_1D_many_parallel(
        xk=xk, fk=fk, nk=nk, xi=xi, fi=fi_parallel,
        sens=None, do_sens=False,
        order=order, knowns=knowns, weighting_method=wm,
        ntasks=4, debug=False,
    )

    assert np.allclose(fi_parallel, fi_serial, atol=1e-14)
