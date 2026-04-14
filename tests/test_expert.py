"""ExpertSolver tests — prepare/solve separation and multi-case fits.

ExpertSolver is the API for the case where many fits share the same
geometry but have different function-value data, so that `prepare()` can
generate and LU-factor the problem matrices once while `solve()` is
called many times with different `fk`. This file verifies that:

  1. prepare/solve recovers the same fi that a single-shot fit_2D/fit_3D
     would.
  2. Running solve() twice on different data updates fi correctly.
  3. ALGO_ITERATIVE does not regress against ALGO_BASIC on an exact
     polynomial fit (both should give the same answer).
"""

import numpy as np

import wlsqm

from conftest import poly2d_order2, poly3d_order2


def _make_expert_2d(ncases, nk_per_case, order=2, algorithm=None, ntasks=1):
    if algorithm is None:
        algorithm = wlsqm.ALGO_BASIC
    nk = np.full(ncases, nk_per_case, dtype=np.int32)
    order_arr = np.full(ncases, order, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)
    return wlsqm.ExpertSolver(
        dimension=2, nk=nk, order=order_arr, knowns=knowns, weighting_method=wm,
        algorithm=algorithm, do_sens=False, ntasks=ntasks, debug=False,
    )


def test_expert_2d_single_case_matches_fit_2d(rng):
    f, fi_expected = poly2d_order2()
    xk = rng.uniform(-1.0, 1.0, size=(30, 2))
    fk = f(xk)
    xi = np.array([0.0, 0.0])

    # Reference via fit_2D.
    fi_ref = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=xi, fi=fi_ref,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )

    # ExpertSolver with a single case.
    es = _make_expert_2d(ncases=1, nk_per_case=30)
    xi_arr = xi.reshape(1, 2)
    xk_arr = xk.reshape(1, 30, 2)
    fk_arr = fk.reshape(1, 30)
    fi_arr = np.zeros((1, wlsqm.number_of_dofs(2, 2)))
    es.prepare(xi=xi_arr, xk=xk_arr)
    es.solve(fk=fk_arr, fi=fi_arr)

    assert np.allclose(fi_arr[0], fi_ref, atol=1e-14)
    assert np.allclose(fi_arr[0], fi_expected, atol=1e-10)


def test_expert_2d_multiple_cases_share_geometry(rng):
    """Same geometry across cases, different fk per case. This is the
    prepare-once, solve-many sweet spot for the expert API.
    """
    f, fi_expected = poly2d_order2()
    ncases = 5
    npts = 25

    # One xi per case (all at the origin), same xk cloud reused for each.
    xi_arr = np.zeros((ncases, 2))
    xk_shared = rng.uniform(-1.0, 1.0, size=(npts, 2))
    xk_arr = np.broadcast_to(xk_shared, (ncases, npts, 2)).copy()

    # Each case fits the same function, so fk is also the same — just a
    # sanity check that the multi-case pipeline doesn't get them mixed up.
    fk_shared = f(xk_shared)
    fk_arr = np.broadcast_to(fk_shared, (ncases, npts)).copy()

    fi_arr = np.zeros((ncases, wlsqm.number_of_dofs(2, 2)))

    es = _make_expert_2d(ncases=ncases, nk_per_case=npts)
    es.prepare(xi=xi_arr, xk=xk_arr)
    es.solve(fk=fk_arr, fi=fi_arr)

    for j in range(ncases):
        assert np.allclose(fi_arr[j], fi_expected, atol=1e-10)


def test_expert_prepare_once_solve_twice(rng):
    """The expert API's raison d'être — factor A once, reuse for many solves."""
    f1, fi1_expected = poly2d_order2()
    # Second case: just use f1 shifted by a constant
    f2 = lambda xy: f1(xy) + 7.5  # noqa: E731
    fi2_expected = fi1_expected.copy()
    fi2_expected[wlsqm.i2_F] += 7.5

    npts = 30
    xi_arr = np.zeros((1, 2))
    xk = rng.uniform(-1.0, 1.0, size=(1, npts, 2))

    es = _make_expert_2d(ncases=1, nk_per_case=npts)
    es.prepare(xi=xi_arr, xk=xk)

    fi_out = np.zeros((1, wlsqm.number_of_dofs(2, 2)))

    # First solve with the original data.
    fk1 = f1(xk[0]).reshape(1, npts)
    es.solve(fk=fk1, fi=fi_out)
    assert np.allclose(fi_out[0], fi1_expected, atol=1e-10)

    # Second solve with the shifted data — without calling prepare() again.
    fk2 = f2(xk[0]).reshape(1, npts)
    es.solve(fk=fk2, fi=fi_out)
    assert np.allclose(fi_out[0], fi2_expected, atol=1e-10)


def test_expert_algo_iterative_matches_algo_basic_on_exact_poly(rng):
    """For an exact polynomial fit, iterative refinement has no work to do —
    the basic solve already gives the right answer. Both algorithms must
    agree to machine precision on this input."""
    f, fi_expected = poly2d_order2()
    npts = 30
    xk = rng.uniform(-1.0, 1.0, size=(1, npts, 2))
    fk = f(xk[0]).reshape(1, npts)
    xi = np.zeros((1, 2))

    es_basic = _make_expert_2d(ncases=1, nk_per_case=npts,
                               algorithm=wlsqm.ALGO_BASIC)
    es_basic.prepare(xi=xi, xk=xk)
    fi_basic = np.zeros((1, wlsqm.number_of_dofs(2, 2)))
    es_basic.solve(fk=fk, fi=fi_basic)

    es_iter = _make_expert_2d(ncases=1, nk_per_case=npts,
                              algorithm=wlsqm.ALGO_ITERATIVE)
    es_iter.prepare(xi=xi, xk=xk)
    fi_iter = np.zeros((1, wlsqm.number_of_dofs(2, 2)))
    es_iter.solve(fk=fk, fi=fi_iter)

    assert np.allclose(fi_basic, fi_iter, atol=1e-12)
    assert np.allclose(fi_basic[0], fi_expected, atol=1e-10)


def test_expert_3d_single_case(rng):
    f, fi_expected = poly3d_order2()
    npts = 40
    xk = rng.uniform(-1.0, 1.0, size=(1, npts, 3))
    fk = f(xk[0]).reshape(1, npts)
    xi = np.zeros((1, 3))

    nk = np.array([npts], dtype=np.int32)
    order = np.array([2], dtype=np.int32)
    knowns = np.zeros(1, dtype=np.int64)
    wm = np.array([wlsqm.WEIGHT_UNIFORM], dtype=np.int32)
    es = wlsqm.ExpertSolver(
        dimension=3, nk=nk, order=order, knowns=knowns, weighting_method=wm,
        algorithm=wlsqm.ALGO_BASIC, do_sens=False, ntasks=1, debug=False,
    )
    es.prepare(xi=xi, xk=xk)
    fi = np.zeros((1, wlsqm.number_of_dofs(3, 2)))
    es.solve(fk=fk, fi=fi)
    assert np.allclose(fi[0], fi_expected, atol=1e-10)
