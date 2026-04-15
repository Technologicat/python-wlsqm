# Changelog

## v1.0.0 (15 April 2026)

First release under modern Python and modern packaging.

Python 2.7 and 3.4 are no longer supported.

### New

- **Python 3.11 – 3.14 supported.**
  - Pre-built wheels on PyPI for Linux, macOS, and Windows, with OpenMP
    parallelism enabled in every wheel.

### Fixed

- **Data race in `fit_1D_many_parallel`**, pre-existing from 2016.
  - Every OpenMP worker clobbered thread-0's work buffer, producing silently
  wrong fits whenever the parallel 1D many-case path ran with `ntasks > 1`. The
  2D/3D branch, the iterative parallel variant, and the serial variant were
  always correct.

- **`rescale_dgeequ` no longer silently accepts singular matrices.**
  - It now checks LAPACK's `info` return and raises `numpy.linalg.LinAlgError`
  when a row or column is exactly zero.

### Changed

- **Installation is now `pip install wlsqm`.**
  - The old `python setup.py install` path is gone; `setup.py` has been removed.
  - The build system is [meson-python](https://meson-python.readthedocs.io/),
  and dev environments are managed with [PDM](https://pdm-project.org/).

- **Language change on "Taylor series."**
  - The package's internal storage layout still uses the same slots a Taylor
  expansion would (e.g. in 2D, `f`, `∂f/∂x`, `∂f/∂y`, `(1/2!) ∂²f/∂x²`,
  `∂²f/∂x∂y`, `(1/2!) ∂²f/∂y²`, …), but the comments and docstrings no longer
  call the model a "Taylor series."
  - In the WLSQM method, the coefficients actually come from a least-squares
  fit, not from analytic differentiation. The error behavior is much better than
  Taylor truncation would predict.
  - The internal C-API function names `taylor_1D/2D/3D` are kept for backwards
  compatibility of downstream `cimport`s — see
  [`wlsqm/fitter/polyeval.pyx`](wlsqm/fitter/polyeval.pyx).

### Internal

- **Port from Cython 0.29 to Cython 3.x**.

- **Comprehensive pytest suite.** 57 tests, covering:
  - polynomial recovery across dimensions and orders,
  - `ExpertSolver` prepare/solve round-trips,
  - interpolation accuracy at interior points,
  - parallel ≡ serial implementation equivalence,
  - finite-difference stencil reproduction,
  - first-derivative robustness to Gaussian noise,
  - edge cases,
  - the LAPACK driver layer, and
  - `.pxd` installability for downstream `cimport` users.

- `ScalingAlgo` is now a proper `enum.IntEnum`, replacing the old bare-class
  Python 2 workaround.

- **GitHub Actions CI**: lint (ruff + cython-lint), test matrix
  (3 OSes × 4 Python versions), cibuildwheel for Linux/macOS/Windows
  wheels, meson-python sdist, and auto-publishing of releases on PyPI.


# Pre-v1.0 history (2016-2017)

## [v0.1.5]
 - support both Python 3.4 and 2.7

## [v0.1.4]
 - actually use the shorter short description (oops)

## [v0.1.3]
 - setup.py is now Python 3 compatible (but wlsqm itself is not yet!)
 - fixed sdist: package also CHANGELOG.md

## [v0.1.2]
 - set zip_safe to False to better work with Cython (important for libs that depend on this one)

## [v0.1.1]
 - change distribution system from distutils to setuptools

## [v0.1.0]
  - initial version
