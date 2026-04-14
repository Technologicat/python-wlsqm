"""Package-level smoke tests: import graph, version, enum dispatch."""

import re

import numpy as np
import pytest

import wlsqm
from wlsqm.utils.lapackdrivers import ScalingAlgo


def test_version_looks_like_pep440():
    v = wlsqm.__version__
    assert isinstance(v, str) and v, "version must be a non-empty string"
    # PEP 440 permissive check: MAJOR.MINOR.PATCH with optional .devN / aN / rcN / post
    assert re.match(r"^\d+\.\d+\.\d+(\.(dev|a|b|rc|post)\d+)?$", v), (
        f"version {v!r} does not look PEP-440 shaped"
    )


def test_submodules_importable():
    from wlsqm.fitter import defs, expert, impl, infra, interp, polyeval, simple  # noqa: F401
    from wlsqm.utils import lapackdrivers, ptrwrap  # noqa: F401


def test_public_reexports_present():
    # Names re-exported from wlsqm.fitter.simple / defs / expert via __init__.py.
    for name in ("fit_1D", "fit_2D", "fit_3D",
                 "fit_1D_many_parallel", "fit_2D_many_parallel", "fit_3D_many_parallel",
                 "ExpertSolver",
                 "WEIGHT_UNIFORM", "WEIGHT_CENTER",
                 "ALGO_BASIC", "ALGO_ITERATIVE",
                 "number_of_dofs"):
        assert hasattr(wlsqm, name), f"wlsqm.{name} missing after star-import"


def test_scaling_algo_is_intenum():
    # Converted from a bare Python class to enum.IntEnum in the modernization
    # pass (Python 2 compatibility workaround is gone).
    import enum
    assert issubclass(ScalingAlgo, enum.IntEnum)
    assert int(ScalingAlgo.ALGO_DGEEQU) == 6
    # Dispatcher in do_rescale compares algo == ScalingAlgo.ALGO_*, so member
    # equality against plain ints must still work.
    assert ScalingAlgo.ALGO_TWOPASS == 3
    assert 3 == ScalingAlgo.ALGO_TWOPASS


def test_number_of_dofs_shapes():
    # 1D: order 0 -> 1, order 1 -> 2, order 2 -> 3, order 3 -> 4, order 4 -> 5
    assert [wlsqm.number_of_dofs(1, k) for k in range(5)] == [1, 2, 3, 4, 5]
    # 2D: 1, 3, 6, 10, 15 (the triangular numbers)
    assert [wlsqm.number_of_dofs(2, k) for k in range(5)] == [1, 3, 6, 10, 15]
    # 3D: 1, 4, 10, 20, 35
    assert [wlsqm.number_of_dofs(3, k) for k in range(5)] == [1, 4, 10, 20, 35]
