## Changelog

### v1.0.0

First release under modern Python and modern packaging. Python 2.7 and 3.4
are no longer supported.

#### New

- **Python 3.11 – 3.14 supported.** Pre-built wheels on PyPI for Linux,
  macOS, and Windows, across all four Python versions, with OpenMP
  parallelism enabled in every wheel.
- **Finite-difference stencil reproduction.** A WLSQM fit on a classical
  central-difference stencil (3-point 1D, 5-point plus 2D, 7-point plus 3D)
  now reproduces the hand-coded stencil result to machine precision on any
  smooth input, not just polynomials. This is the natural generalization of
  WLSQM's polynomial-recovery property and is pinned by the test suite.

#### Fixed

- **Data race in `fit_1D_many_parallel`**, pre-existing from 2016. The 1D
  branch of the basic parallel many-case fitter passed a compile-time
  constant `TASKID = 0` to `impl.solve()` instead of the per-thread
  `taskid = openmp.omp_get_thread_num()`. Every OpenMP worker clobbered
  thread-0's work buffer, producing silently wrong fits whenever the
  parallel 1D many-case path ran with `ntasks > 1`. The 2D/3D branch, the
  iterative parallel variant, and the serial variant were always correct.
  A regression test (64 cases × 4 threads) now pins this.
- **`rescale_dgeequ` no longer silently accepts singular matrices.** It now
  checks LAPACK's `info` return and raises `numpy.linalg.LinAlgError` when
  a row or column is exactly zero, instead of returning nonsense scaling
  factors that would poison the downstream solve.

#### Changed

- **Installation is now `pip install wlsqm`.** The old `python setup.py
  install` path is gone; `setup.py` has been removed. The build system is
  [meson-python](https://meson-python.readthedocs.io/), and dev environments
  are managed with [PDM](https://pdm-project.org/).
- **Language change on "Taylor series."** The package's internal storage
  layout still uses the same slots a Taylor expansion would (function
  value, first derivatives, second derivatives divided by `2!`, …), but
  the comments and docstrings no longer call the model a "Taylor series."
  The coefficients come from a least-squares fit, not from analytic
  differentiation, and the error behavior is much better than Taylor
  truncation would predict. The internal C-API function names
  `taylor_1D/2D/3D` are kept for backwards compatibility of downstream
  `cimport`s — see [`wlsqm/fitter/polyeval.pyx`](wlsqm/fitter/polyeval.pyx).
- **Comprehensive pytest suite.** 57 tests covering polynomial recovery
  across dimensions and orders, `ExpertSolver` prepare/solve round-trips,
  interpolation accuracy at interior points, parallel ≡ serial equivalence,
  finite-difference stencil reproduction, first-derivative robustness to
  Gaussian noise, edge cases, the LAPACK driver layer, and `.pxd`
  installability for downstream `cimport` users.

#### Internal

- Port from Cython 0.29 to Cython 3.x. `noexcept` audit on every `cdef
  ... nogil` function, split between pure computational helpers
  (`noexcept`) and LAPACK wrappers / fit dispatchers (`except -1`). `fma`
  now imported from `libc.math` instead of a manual `cdef extern` hack
  that worked around a long-fixed bug in Cython 0.20.1.
- All `DEF` compile-time constants replaced with module-level `cdef`
  constants or inlined as literals at call sites (Cython 3 deprecated
  `DEF`). Function-local protocol constants like `TASKID`, `NTASKS`, and
  `MODE_BASIC` / `MODE_ITERATIVE` live at module scope in `simple.pyx`
  where the value is a project-wide convention, and inside each function
  where the value is per-function.
- `ScalingAlgo` is now a proper `enum.IntEnum`, replacing the old bare-
  class Python 2 workaround.
- GitHub Actions CI: lint (ruff + cython-lint), test matrix (3 OSes × 4
  Python versions), cibuildwheel for Linux/macOS/Windows wheels,
  meson-python sdist, and trusted-publisher PyPI publishing on `v*` tag
  push.
- Copyright updated to 2016–2026 and affiliation updated to JAMK
  University of Applied Sciences.


## Pre-v1.0 history

### [v0.1.5]
 - support both Python 3.4 and 2.7

### [v0.1.4]
 - actually use the shorter short description (oops)

### [v0.1.3]
 - setup.py is now Python 3 compatible (but wlsqm itself is not yet!)
 - fixed sdist: package also CHANGELOG.md

### [v0.1.2]
 - set zip_safe to False to better work with Cython (important for libs that depend on this one)

### [v0.1.1]
 - change distribution system from distutils to setuptools

### [v0.1.0]
  - initial version
