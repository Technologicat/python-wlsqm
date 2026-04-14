"""Tests for wlsqm.utils.lapackdrivers — the thin LAPACK wrapper layer.

These aren't LAPACK correctness tests (SciPy has those); they verify that
our Python-facing wrappers accept the expected array layouts, return the
right shapes, and dispatch through the (ex-class-now-IntEnum) `ScalingAlgo`
correctly. The previously-silent `rescale_dgeequ_c` info-ignoring bug also
gets a regression test here.
"""

import numpy as np
import pytest

from wlsqm.utils.lapackdrivers import (
    ScalingAlgo,
    do_rescale,
    general,
    rescale_columns,
    rescale_dgeequ,
    rescale_ruiz2001,
    rescale_twopass,
    symmetric,
    tridiag,
)


def test_tridiag_4x4():
    # T = [[ 2, -1,  0, 0],
    #      [-1,  2, -1, 0],
    #      [ 0, -1,  2,-1],
    #      [ 0,  0, -1, 2]]
    # b = [1, 0, 0, 1] -> x = [0.625, 0.25, 0.5, 0.75] (verified via np.linalg.solve)
    a = np.array([0.0, -1.0, -1.0, -1.0])  # sub-diagonal (a[0] unused)
    b_diag = np.array([2.0, 2.0, 2.0, 2.0])
    c = np.array([-1.0, -1.0, -1.0, 0.0])  # super-diagonal (c[n-1] unused)
    x = np.array([1.0, 0.0, 0.0, 1.0])     # overwritten in place with solution
    tridiag(a, b_diag, c, x)
    expected = np.array([0.625, 0.25, 0.5, 0.75])
    assert np.allclose(x, expected, atol=1e-14)


def test_general_solve_matches_numpy(rng):
    n = 5
    A_dense = rng.standard_normal((n, n))
    b_dense = rng.standard_normal(n)
    expected = np.linalg.solve(A_dense, b_dense)

    # LAPACK wrapper mutates A and b in place; needs Fortran layout.
    A = np.asfortranarray(A_dense.copy())
    b = b_dense.copy()
    general(A, b)
    assert np.allclose(b, expected, atol=1e-12)


def test_symmetric_solve_matches_numpy(rng):
    n = 5
    M = rng.standard_normal((n, n))
    A_dense = (M + M.T) / 2.0 + n * np.eye(n)  # symmetric + diagonally dominant
    b_dense = rng.standard_normal(n)
    expected = np.linalg.solve(A_dense, b_dense)

    A = np.asfortranarray(A_dense.copy())
    b = b_dense.copy()
    symmetric(A, b)
    assert np.allclose(b, expected, atol=1e-12)


def test_rescale_columns_is_unit_column_norm(rng):
    n = 4
    A_dense = rng.standard_normal((n, n)) * 100.0
    A = np.asfortranarray(A_dense.copy())
    _row, _col = rescale_columns(A)
    # After column-euclidean scaling, each column of A has unit L2 norm.
    col_norms = np.linalg.norm(A, axis=0)
    assert np.allclose(col_norms, 1.0, atol=1e-12)


def test_rescale_twopass_matches_do_rescale_dispatch(rng):
    n = 4
    A_dense = rng.standard_normal((n, n)) * 100.0
    # Two ways to reach the same algorithm.
    A1 = np.asfortranarray(A_dense.copy())
    A2 = np.asfortranarray(A_dense.copy())
    r1, c1 = rescale_twopass(A1)
    r2, c2 = do_rescale(A2, ScalingAlgo.ALGO_TWOPASS)
    assert np.allclose(np.asarray(r1), np.asarray(r2), atol=1e-14)
    assert np.allclose(np.asarray(c1), np.asarray(c2), atol=1e-14)
    assert np.allclose(A1, A2, atol=1e-14)


def test_rescale_ruiz2001_preserves_symmetry(rng):
    n = 4
    M = rng.standard_normal((n, n))
    A_sym = (M + M.T) / 2.0 + n * np.eye(n)
    A = np.asfortranarray(A_sym.copy())
    rescale_ruiz2001(A)
    # Ruiz's iterative algorithm scales rows and columns by the SAME factors
    # in the symmetric case, so A must remain symmetric.
    assert np.allclose(A, A.T, atol=1e-12)


def test_rescale_dgeequ_well_conditioned_succeeds():
    A = np.asfortranarray(np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64))
    row_scale, col_scale = rescale_dgeequ(A.copy(order="F"))
    # DGEEQU for a well-conditioned matrix returns strictly positive factors.
    assert np.all(np.asarray(row_scale) > 0.0)
    assert np.all(np.asarray(col_scale) > 0.0)


def test_rescale_dgeequ_singular_raises_linalgerror():
    """Regression: `rescale_dgeequ_c` used to ignore DGEEQU's `info` output
    and silently return success even on singular rows/columns. Now we must
    get a LinAlgError."""
    A = np.asfortranarray(np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64))
    with pytest.raises(np.linalg.LinAlgError):
        rescale_dgeequ(A)


def test_do_rescale_unknown_algo_raises():
    A = np.asfortranarray(np.eye(3, dtype=np.float64))
    with pytest.raises(ValueError, match="Unknown algorithm"):
        do_rescale(A, 999)
