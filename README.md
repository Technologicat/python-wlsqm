# wlsqm

![top language](https://img.shields.io/github/languages/top/Technologicat/python-wlsqm)
![supported Python versions](https://img.shields.io/pypi/pyversions/wlsqm)
![supported implementations](https://img.shields.io/pypi/implementation/wlsqm)
![CI status](https://img.shields.io/github/actions/workflow/status/Technologicat/python-wlsqm/ci.yml?branch=master)

![version on PyPI](https://img.shields.io/pypi/v/wlsqm)
![PyPI package format](https://img.shields.io/pypi/format/wlsqm)
![dependency status](https://img.shields.io/librariesio/github/Technologicat/python-wlsqm)

![license](https://img.shields.io/pypi/l/wlsqm)
![open issues](https://img.shields.io/github/issues/Technologicat/python-wlsqm)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](http://makeapullrequest.com/)

Fast and accurate weighted least squares meshless interpolator for Python.

We use [semantic versioning](https://semver.org/).

For my stance on AI contributions, see the [collaboration guidelines](https://github.com/Technologicat/substrate-independent/blob/main/collaboration.md).

![2D example](example.png)


## Introduction

WLSQM (Weighted Least SQuares Meshless) constructs a piecewise polynomial surrogate model on scattered data: given scalar values at a point cloud in 1D, 2D, or 3D, it fits a local polynomial (up to 4th order) in the neighborhood of each query point, by weighted least squares. From the surrogate you can read off the function value and any derivative up to the polynomial order, at the fit origin or anywhere inside the local neighborhood.

Use cases:

- **Numerical differentiation** of data known only as values at discrete points. Applies to explicit algorithms for IBVPs: compute a gradient or Laplacian on a scattered point cloud without needing a mesh.
- **Response-surface modeling** on an unstructured set of design points.
- **Smoothing noisy function values** — the averaging effect of the least-squares fit denoises first derivatives significantly. (Second derivatives are more sensitive to noise; for those, the robust recipe is to run WLSQM once to recover first derivatives on the neighborhood, then run it again on those to get second derivatives.)

No grid or mesh is needed. The only restriction on geometry is non-degeneracy — e.g. 2D points must not all fall on the same 1D line.

### Not a Taylor series

Despite the storage layout looking like a Taylor expansion (slots for `f`, `df/dx`, `df/dy`, `d²f/dx²`, ..., indexed by monomial degree), WLSQM is **not** evaluating a Taylor series. The polynomial is fitted over a local neighborhood of data points by weighted least squares; the coefficients come from a linear solve, not from differentiating an analytic function at the origin. The error behavior is correspondingly much better than Taylor truncation error would predict, thanks to the averaging effect of the least-squares fit.

This method appears in the literature under several names — MLS (moving least squares), WLSQM (weighted least squares meshless), "diffuse approximation." These are essentially the same idea with varying weighting function choices.

### Academic reference

This is an independent implementation of the algorithm described (in the 2nd-order 2D case) in section 2.2.1 of Hong Wang (2012), *Evolutionary Design Optimization with Nash Games and Hybridized Mesh/Meshless Methods in Computational Fluid Dynamics*, Jyväskylä Studies in Computing 162, University of Jyväskylä. [ISBN 978-951-39-5007-1 (PDF)](http://urn.fi/URN:ISBN:978-951-39-5007-1)

Full theory for the generalized version (including the case of unknown function values, and the accuracy analysis): `doc/wlsqm_gen.pdf` in this repository.


## Features

- **1D / 2D / 3D point clouds**, polynomials of order 0 through 4.
- **Any derivative** up to the polynomial order is available: at the fit origin as a DOF of the linear solve, and at any point inside the neighborhood via the interpolator. Differentiation of the basis polynomials is hardcoded for speed.
- **Knowns.** At the fit origin, any subset of `{f, ∂f/∂x, ∂f/∂y, ∂²f/∂x², …}` can be marked as known. The known DOFs are eliminated algebraically, shrinking the linear system to just the unknowns. The function value itself may be unknown — useful e.g. for Neumann boundary conditions in a PDE solver.
- **Weighting methods.** `WEIGHT_UNIFORM` gives the best overall fit of function values. `WEIGHT_CENTER` emphasizes points near the fit origin, improving derivative estimates there at the cost of fit quality far from the origin.
- **Sensitivity data.** Solution DOFs can be optionally differentiated w.r.t. the input function values, so you know how the output moves when the input wiggles.
- **Expert mode with separate prepare and solve stages.** If many fits share the same geometry but have different function-value data, `ExpertSolver.prepare()` generates and LU-factors the problem matrices once, and `solve()` can then be called many times with different `fk`. This is the fast path for time-stepping an IBVP over a fixed point cloud.
- **Parallel across independent local problems.** `fit_*_many_parallel` and `ExpertSolver(ntasks=N)` run independent fits across OpenMP threads. The linear-solver step is also parallel.
- **Speed.**
  - Performance-critical code is in Cython with the GIL released.
  - LAPACK is called directly via [SciPy's Cython-level bindings](https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html); no GIL round-trip for the solver loop.
  - The polynomial evaluator uses a symmetric Horner-like form with fused multiply-add (`fma`). See [`wlsqm/fitter/polyeval.pyx`](wlsqm/fitter/polyeval.pyx).
- **Accuracy.**
  - Problem matrices are preconditioned by a symmetry-preserving iterative scaling (Ruiz 2001) before LU factorization, which is critical for high-order fits.
  - Optional iterative refinement inside the solver mitigates roundoff further.
  - FMA in the polynomial evaluator rounds only once per accumulation step.


## Examples

Minimal 2D fit:

```python
import numpy as np
import wlsqm

# Fit f(x,y) = 1 + 2x + 3y + 4xy + 5x² + 6y² on a scattered point cloud
# centered at the origin.
rng = np.random.default_rng(42)
xk = rng.uniform(-1.0, 1.0, size=(30, 2))
fk = 1 + 2*xk[:,0] + 3*xk[:,1] + 4*xk[:,0]*xk[:,1] + 5*xk[:,0]**2 + 6*xk[:,1]**2

xi = np.array([0.0, 0.0])
fi = np.zeros(wlsqm.number_of_dofs(2, 2))
wlsqm.fit_2D(
    xk=xk, fk=fk, xi=xi, fi=fi,
    sens=None, do_sens=False,
    order=2, knowns=0,
    weighting_method=wlsqm.WEIGHT_UNIFORM,
    debug=False,
)

# fi now holds the partial derivatives at (0, 0) in the order
# F, X, Y, X2, XY, Y2. For this exact polynomial, the fit recovers
# [1, 2, 3, 10, 4, 12] to machine precision.
print(fi)
```

For a comprehensive tour of the API, see [`examples/wlsqm_example.py`](examples/wlsqm_example.py). For a minimal `ExpertSolver` example that demonstrates the prepare/solve separation, see [`examples/expertsolver_example.py`](examples/expertsolver_example.py).


## Installation

### From PyPI

```bash
pip install wlsqm
```

Pre-built wheels are available for Linux, macOS, and Windows, for Python 3.11–3.14. Parallel OpenMP is enabled in all published wheels:

- **Linux:** GCC's `libgomp` via manylinux.
- **macOS:** LLVM's `libomp` (from conda-forge) bundled into the wheel by `delocate-wheel`.
- **Windows:** MSVC's `vcomp140.dll`, which ships with every Python-for-Windows install.

### From source

```bash
git clone https://github.com/Technologicat/python-wlsqm.git
cd python-wlsqm
pip install .
```

For maximum performance on your specific machine, build with architecture-specific optimizations:

```bash
CFLAGS="-march=native" pip install --no-build-isolation .
```

PyPI wheels use generic `-O2` because `-march=native` bakes the build machine's instruction set into the binary — a wheel built with AVX-512 would crash on a CPU without it. Building from source avoids that and lets the compiler target your specific hardware.

### macOS: OpenMP for source builds

macOS's Apple Clang does not ship an OpenMP runtime. For a source install to produce a parallel build, install `libomp` first:

```bash
brew install libomp
pip install --no-binary wlsqm --no-build-isolation wlsqm
```

Without `libomp`, the source build fails at compile time because Cython's `cimport openmp` in the source emits `#include <omp.h>`. The published wheels from PyPI already bundle their own `libomp.dylib` via `delocate-wheel`, so `pip install wlsqm` (without `--no-binary`) works on macOS regardless of whether you have Homebrew's libomp.

### Development setup

Uses [meson-python](https://meson-python.readthedocs.io/) as the build backend and [PDM](https://pdm-project.org/) for dependency management:

```bash
git clone https://github.com/Technologicat/python-wlsqm.git
cd python-wlsqm
pdm config venv.in_project true
pdm use 3.14                             # or whichever Python you prefer
pdm install                              # creates .venv, installs dev deps
export PATH="$(pwd)/.venv/bin:$PATH"     # meson and ninja must be on PATH
pip install --no-build-isolation -e .    # editable install
```

After editing a `.pyx` or `.pxd` file, the next `import wlsqm` auto-rebuilds the changed extension. No manual reinstall needed.

`--no-build-isolation` is required for editable installs with meson-python: the on-import rebuild mechanism needs Cython, NumPy, meson, and ninja to remain available in the venv, not just in a throwaway PEP 517 overlay.

### Running the tests

```bash
pdm run pytest tests/ -v
```

57 tests covering: polynomial recovery (1D/2D/3D, order 0–4), `ExpertSolver` prepare/solve round-trips, interpolation accuracy, `_many_parallel` ≡ `_many` (serial) equivalence, classical finite-difference-stencil equivalence on non-polynomial inputs (sin, exp, Gaussian, …), first-derivative robustness to Gaussian noise, the LAPACK driver layer, and `.pxd` install verification.


## Documentation

- **API:** docstrings in [`wlsqm/fitter/simple.pyx`](wlsqm/fitter/simple.pyx) (simple API) and [`wlsqm/fitter/expert.pyx`](wlsqm/fitter/expert.pyx) (`ExpertSolver`).
- **Examples:** [`examples/wlsqm_example.py`](examples/wlsqm_example.py) for the full tour, [`examples/expertsolver_example.py`](examples/expertsolver_example.py) for `ExpertSolver` specifically.
- **Theory PDFs** in [`doc/`](doc/):
  - [`wlsqm_gen.pdf`](doc/wlsqm_gen.pdf) — the generalized version (including the case of unknown function values), the accuracy analysis, and why WLSQM works.
  - [`wlsqm.pdf`](doc/wlsqm.pdf) — older writeup for the pure-Python version originally written for FREYA, plus the sensitivity calculation.
  - [`eulerflow.pdf`](doc/eulerflow.pdf) — application example in compressible flow, with a cleaner presentation of the original version.

See [TODO.md](TODO.md) for known gaps in the theory PDFs.


## Dependencies

- [NumPy](http://www.numpy.org) ≥ 1.25
- [SciPy](http://www.scipy.org) ≥ 1.9 (both build-time for `cython_lapack` and runtime)
- [Cython](http://www.cython.org) ≥ 3.0 (build-time only)
- OpenMP (build-time; see platform notes above)

Requires Python ≥ 3.11.


## License

[BSD 2-Clause](LICENSE.md). Copyright 2016–2026 Juha Jeronen, University of Jyväskylä, and JAMK University of Applied Sciences.


#### Acknowledgement

The original version of this work, in 2016–2017, was financially supported by the Jenny and Antti Wihuri Foundation.
