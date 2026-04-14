"""Scratchpad of smoke tests accumulated during the modernization pass.

This is NOT a pytest file yet — the functions take no fixtures and the checks
are plain `assert` / print statements. Phase 6 will turn these into proper
`test_*.py` modules with parametrization and tolerances, but until then,
running this script is the fastest way to re-verify that a build still
matches analytical derivatives on a handful of known polynomials.

    python tests/_smoke_seeds.py
"""

import numpy as np

import wlsqm
from wlsqm.utils.lapackdrivers import (
    ScalingAlgo,
    rescale_dgeequ,
    tridiag,
)


# The 2D DOF ordering (see defs.pxd) is F, X, Y, X2, XY, Y2 for order=2.
# For f(x,y) = 1 + 2x + 3y + 4xy + 5x**2 + 6y**2 the derivatives at (0,0) are:
#   F = 1, X = 2, Y = 3, XY = 4, X2 = d2f/dx2 = 10, Y2 = d2f/dy2 = 12
# wlsqm stores the PARTIAL DERIVATIVES (not monomial coefficients), so X2 is
# 2 * 5 = 10 and Y2 is 2 * 6 = 12.
EXPECTED_2D = np.array([1., 2., 3., 10., 4., 12.])


def smoke_2d_basic_fit():
    rng = np.random.default_rng(42)
    xk = rng.uniform(-1, 1, size=(30, 2))
    fk = (
        1
        + 2 * xk[:, 0]
        + 3 * xk[:, 1]
        + 4 * xk[:, 0] * xk[:, 1]
        + 5 * xk[:, 0] ** 2
        + 6 * xk[:, 1] ** 2
    )
    xi = np.array([0.0, 0.0])
    fi = np.zeros(wlsqm.number_of_dofs(2, 2))
    wlsqm.fit_2D(
        xk=xk, fk=fk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    assert np.allclose(fi, EXPECTED_2D, atol=1e-12), f"2D fit: {fi} vs {EXPECTED_2D}"
    print("smoke_2d_basic_fit: OK", fi)


def smoke_3d_basic_fit():
    rng = np.random.default_rng(42)
    xk = rng.uniform(-1, 1, size=(40, 3))
    fk = (
        1 + 2 * xk[:, 0] - xk[:, 1] + 3 * xk[:, 2]
        + xk[:, 0] * xk[:, 1]
    )
    xi = np.array([0.0, 0.0, 0.0])
    fi = np.zeros(wlsqm.number_of_dofs(3, 2))
    wlsqm.fit_3D(
        xk=xk, fk=fk, xi=xi, fi=fi,
        sens=None, do_sens=False,
        order=2, knowns=0,
        weighting_method=wlsqm.WEIGHT_UNIFORM,
        debug=False,
    )
    # F, X, Y, Z, X2, XY, Y2, YZ, Z2, XZ
    assert abs(fi[0] - 1.0) < 1e-10 and abs(fi[1] - 2.0) < 1e-10 and abs(fi[2] + 1.0) < 1e-10 and abs(fi[3] - 3.0) < 1e-10, f"3D: {fi[:4]}"
    print("smoke_3d_basic_fit: OK", fi[:6])


def smoke_parallel_1d_many_cases():
    """Regression test for the data-race bug in generic_fit_basic_many_parallel's
    1D branch (old simple.pyx passed TASKID=0 instead of `taskid` from
    openmp.omp_get_thread_num() to impl.solve, so all threads clobbered work
    buffer 0). 64 cases * 4 threads should recover every per-case line fit.
    """
    rng = np.random.default_rng(42)
    ncases = 64
    nk_each = 25
    xk_1d = rng.uniform(-1, 1, size=(ncases, nk_each))
    fk_1d = np.stack([j + (j + 1) * xk_1d[j] for j in range(ncases)])
    xi_1d = np.zeros(ncases)
    fi_out = np.zeros((ncases, wlsqm.number_of_dofs(1, 1)))
    order = np.ones(ncases, dtype=np.int32)
    knowns = np.zeros(ncases, dtype=np.int64)
    wm = np.full(ncases, wlsqm.WEIGHT_UNIFORM, dtype=np.int32)
    nk_arr = np.full(ncases, nk_each, dtype=np.int32)
    wlsqm.fit_1D_many_parallel(
        xk=xk_1d, fk=fk_1d, nk=nk_arr, xi=xi_1d, fi=fi_out,
        sens=None, do_sens=False, order=order, knowns=knowns,
        weighting_method=wm, ntasks=4, debug=False,
    )
    for j in range(ncases):
        assert abs(fi_out[j, 0] - j) < 1e-10, f"case {j}: F got {fi_out[j, 0]}"
        assert abs(fi_out[j, 1] - (j + 1)) < 1e-10, f"case {j}: X got {fi_out[j, 1]}"
    print(f"smoke_parallel_1d_many_cases: OK {ncases} cases x 4 threads")


def smoke_rescale_dgeequ_raises_on_singular():
    """Regression for the previously-silent bug where rescale_dgeequ_c
    ignored DGEEQU's `info` output and always returned 1 on success."""
    A = np.asfortranarray(np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64))
    try:
        rescale_dgeequ(A)
    except np.linalg.LinAlgError:
        print("smoke_rescale_dgeequ_raises_on_singular: OK")
    else:
        raise AssertionError("expected LinAlgError on singular row")


def smoke_rescale_dgeequ_ok():
    A = np.asfortranarray(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    r, c = rescale_dgeequ(A.copy(order="F"))
    print("smoke_rescale_dgeequ_ok:", np.asarray(r), np.asarray(c))


def smoke_scaling_algo_is_intenum():
    assert int(ScalingAlgo.ALGO_DGEEQU) == 6
    # comparison against plain int still works (dispatcher relies on this):
    assert ScalingAlgo.ALGO_TWOPASS == 3
    print("smoke_scaling_algo_is_intenum: OK")


def smoke_tridiag():
    # Solve a 4x4 tridiagonal system by hand: T x = b
    # T = [[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]], b = [1,0,0,1]
    a = np.array([0.0, -1.0, -1.0, -1.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    c = np.array([-1.0, -1.0, -1.0, 0.0])
    x = np.array([1.0, 0.0, 0.0, 1.0])
    tridiag(a, b, c, x)
    expected = np.array([0.625, 0.25, 0.5, 0.75])  # from direct solve
    assert np.allclose(x, expected, atol=1e-12), f"tridiag: {x} vs {expected}"
    print("smoke_tridiag: OK", x)


def run_all():
    smoke_2d_basic_fit()
    smoke_3d_basic_fit()
    smoke_parallel_1d_many_cases()
    smoke_rescale_dgeequ_ok()
    smoke_rescale_dgeequ_raises_on_singular()
    smoke_scaling_algo_is_intenum()
    smoke_tridiag()
    print("all smoke tests passed")


if __name__ == "__main__":
    run_all()
