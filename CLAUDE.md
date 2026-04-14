# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is wlsqm

Cython library that constructs a piecewise polynomial surrogate model on scattered point-cloud data in 1D, 2D, or 3D, by weighted least squares. From the surrogate you read off the function value and any derivative up to the polynomial order. BSD-2-Clause licensed.

The algorithm fits a local polynomial over the neighborhood of each query point, using the same monomial basis a Taylor expansion would use (so the DOF layout reads as "F, X, Y, X², XY, Y², …"). **It is not a Taylor series.** The coefficients come from a weighted least-squares linear solve, not from analytic differentiation — which is why the error behavior is much better than Taylor truncation would predict, and why the method works at all on noisy data. Downstream code that uses wlsqm should think of the result as "least-squares-optimal local derivative estimates," not "exact analytic derivatives."

The method appears in the literature under several names: MLS (moving least squares), WLSQM (weighted least squares meshless), "diffuse approximation." They are essentially the same idea.

Runtime dependencies are NumPy and SciPy. SciPy is needed at both build time (the Cython `cimport scipy.linalg.cython_lapack` requires the headers at compile time) and runtime (the compiled extensions' `cimport` chain resolves through SciPy's module-level C API at import time).

## Build and Development

Uses meson-python as build backend, PDM for dependency management. Python ≥ 3.11.

```bash
pdm config venv.in_project true
pdm use 3.14
pdm install                                                 # dev deps into .venv
export PATH="$(pwd)/.venv/bin:$PATH"                        # meson / ninja must be on PATH
pip install --no-build-isolation -e .                       # editable install
```

After editing a `.pyx` or `.pxd` file, the next `import wlsqm` auto-rebuilds the changed extension.

**Why `--no-build-isolation`:** meson-python's editable loader rebuilds the extension on import, so it needs `meson`, `ninja`, Cython, NumPy, and SciPy to remain available in the venv — not just in a throwaway PEP 517 overlay. PDM's default `pdm install` runs the backend in an isolated overlay whose `ninja` path gets burned into the loader and then disappears, causing `FileNotFoundError: .../ninja` on import. The `pip install --no-build-isolation -e .` form reuses the venv directly and produces a loader with stable paths.

**Version:** single source of truth is `wlsqm/VERSION`. Read by `meson.build` (build time), `pyproject.toml` (dynamic), and `wlsqm/__init__.py` (runtime). Only edit `wlsqm/VERSION` when bumping.

**OpenMP.** Linux (GCC libgomp), macOS (Apple Clang + Homebrew libomp), and Windows (MSVC vcomp140) are all supported. The meson build uses `dependency('openmp', required: false)` so that a build can in principle proceed without OpenMP — but in practice the `.pyx` sources `cimport openmp` and Cython unconditionally emits `#include <omp.h>`, so `omp.h` must be available at compile time regardless. On macOS, that means `brew install libomp` for source installs; published wheels bundle conda-forge's `libomp.dylib` via `delocate-wheel`.

## Running Tests

```bash
pdm run pytest tests/ -v
```

57 tests covering polynomial recovery (1D/2D/3D, orders 0–4), `ExpertSolver` prepare/solve round-trips, interpolation accuracy, `_many_parallel` ≡ `_many` serial equivalence, classical finite-difference stencil equivalence on non-polynomial inputs, first-derivative robustness to Gaussian noise, the LAPACK driver layer, and `.pxd` installability.

## Architecture

### Simple API vs. Expert API

Two Python-facing APIs on top of the same internal machinery:

- **`wlsqm.fit_1D / fit_2D / fit_3D`** — one-shot fit of a single local model. The `*_many` variants loop over many independent fits. The `*_many_parallel` variants run the loop across OpenMP threads.
- **`wlsqm.ExpertSolver`** — prepare/solve separation. `prepare(xi, xk)` generates and LU-factorizes the problem matrices for every fit in the batch; `solve(fk, fi)` reuses the factored matrices against new function-value data. Fast path for time-stepping an IBVP over a fixed point cloud.

### C structs, not cdef classes

The hot-path data structures — `Allocator`, `BufferSizes`, `CaseManager`, `Case` — are all `cdef struct` in `wlsqm/fitter/infra.pxd`, with C-style constructor/destructor functions (`Case_new`, `Case_del`, `CaseManager_new`, `CaseManager_commit`, `CaseManager_del`, …). The only real `cdef class` in the codebase is `PointerWrapper` in `wlsqm/utils/ptrwrap.pyx`, a trivial void-pointer carrier used in one place by `ExpertSolver`.

**Do not convert the C structs to `cdef class`.** The struct layout is what lets `Case_new` be called from inside `nogil` parallel loops without allocating Python objects.

### The .pxd / .pyx split

Extension modules under `wlsqm/fitter/` and `wlsqm/utils/` come in matched `.pxd` + `.pyx` pairs (except `expert.pyx` which is Python-API-only and has no `.pxd`, and `defs.pxd` which exposes module-level `cdef int` constants rather than function declarations). The `.pxd` is the Cython-level API that downstream `cimport` users consume; the `.pyx` contains the implementation. Both must be installed alongside the compiled `.so` / `.pyd` for `cimport wlsqm.fitter.*` to work from other Cython projects.

### Inter-module cimport graph

Build order (leaf → root):

```
defs, ptrwrap → infra, polyeval → lapackdrivers → interp → impl → simple, expert
```

`defs` is a pure-constants leaf. `lapackdrivers` cimports `scipy.linalg.cython_lapack`. `simple` and `expert` sit at the top of the chain and `cimport` everything below them.

### The `defs` constant system is NOT an enum

`wlsqm/fitter/defs.pxd` declares module-level `cdef int` variables (e.g. `i2_X2_c`, `b2_XY_c`, `ALGO_BASIC_c`); `wlsqm/fitter/defs.pyx` assigns their values and also exports Python-accessible copies (`i2_X2`, `b2_XY`, `ALGO_BASIC`). This **looks** like an enum and the `ALGO_*` / `WEIGHT_*` constants even act like one, but it is not — and must not be converted to either `cdef enum` or Python `enum.Enum`:

- **`i1_*`, `i2_*`, `i3_*`** are **array indices** into the `fi` DOF vector. Their specific numerical values (0, 1, 2, …) are load-bearing: the fitting code writes `fi[ i2_X2_c ] = ...` on the assumption that `i2_X2_c` is a specific integer slot in a dense array whose layout is known to every `make_c_*`, `make_A`, `solve`, `interpolate_*` function.
- **`b1_*`, `b2_*`, `b3_*`** are **bitmasks** for the `knowns` parameter. `b2_F_c = 1 << i2_F_c`. They combine via `|`. An enum would force every bit to fit into a single named value.
- **`SIZE1_c`, `SIZE2_c`, `SIZE3_c`** are array sizes: one-past-end of each order's DOF range.

The one place where an enum conversion **did** make sense was `ScalingAlgo` in `wlsqm/utils/lapackdrivers.pyx`, which is now a proper `enum.IntEnum`. It was a Python 2 compatibility workaround (a bare class with integer class attributes); the old comment literally said "TODO: use real enum type for Python 3.4+".

### Protocol constants in simple.pyx

The serial fitting routines in `wlsqm/fitter/simple.pyx` share a set of protocol constants — `TASKID = 0`, `NTASKS = 1`, and `MODE_BASIC = 0` / `MODE_ITERATIVE = 1` — that live at module scope at the top of the file. They express project-wide conventions ("serial path has a single task at work buffer 0") and are passed as positional arguments to `CaseManager_new` / `Case_new` / `impl.solve`. Do not inline them as bare literals at call sites; the named constants document what the argument position means.

`expert.pyx` has an analogous local `SERIAL_TASKID = 0` inside `ExpertSolver.solve`, used only in the `ntasks == 1` serial-fallback branches of that one method. Local scope is correct there because the parallel and serial paths share a function, and the name `SERIAL_TASKID` self-documents the context at each call site.

### The lapackdrivers layer

`wlsqm/utils/lapackdrivers.pyx` is a thin Cython wrapper over SciPy's Cython LAPACK bindings, exposing:

- Single-matrix solvers (`general`, `symmetric`, `tridiag`, `svd`).
- Batched solvers for many independent systems (`mgeneral_c`, `msymmetric_c`, and their `*p_c` parallel variants that iterate with OpenMP `prange`).
- Six preconditioning / scaling algorithms (`rescale_columns_c`, `rescale_rows_c`, `rescale_twopass_c`, `rescale_dgeequ_c`, `rescale_ruiz2001_c`, `rescale_scalgm_c`), dispatched through a function-pointer table by `do_rescale`. **Keep all six.** They were compared experimentally for wlsqm's own use (Ruiz 2001 won and is the default), and the full set is a legitimate home for the comparison.

The LAPACK wrapper functions return `int` and use `except -1 nogil` to propagate LAPACK errors as Python exceptions. The pure computational helpers (`cimin`, `cimax`, `copygeneral_c`, `symmetrize_c`, `init_scaling_c`, `apply_scaling_c`, the `basic_scale_up/down_*` family) use `noexcept nogil` and never raise. Do not mix the two styles — they are chosen per-function based on whether the function can fail.

### Critical constraint: .pxd installation

Every `.pxd` file must be installed alongside the compiled extension so downstream `cimport wlsqm.fitter.*` works. Handled by `py.install_sources(...)` in each subpackage's `meson.build`. `tests/test_cimport.py` pins this invariant by generating a minimal `.pyx` per module and asking `cython -3` to compile it; the test fails if any `.pxd` is missing or unreachable.

## Linting

**Python files** (ruff, blocking):

```bash
ruff check . --ignore SIM103
```

Plus a non-blocking advisory pass for the return-condition-directly rule:

```bash
ruff check . --select SIM103 || true
```

**Cython files** (cython-lint, non-blocking in CI, blocking-clean in practice):

```bash
cython-lint wlsqm/fitter/*.pyx wlsqm/fitter/*.pxd \
            wlsqm/utils/*.pyx wlsqm/utils/*.pxd
```

Config for both lives in `pyproject.toml`. The canonical lint config and the rationale behind each ignore are in `~/.claude/PROJECT-SETUP-NOTES.md` under "Lint and style configuration."

## Code Conventions

- **Line width:** ~130 for Python, ~200 for Cython signatures (many `cdef` signatures are legitimately long because of memoryview type declarations).
- **Docstring format:** the `.pyx` files use a custom `def name(args):\n"""def name(args):\n\n...` header-echo convention. Leave existing docstrings in that style.
- **Comments can carry math.** Derivations, accuracy-bound sketches, and back-of-the-envelope FLOP counts in comments are the project style. Don't prune them.
- **Dependencies:** NumPy and SciPy are the runtime deps. OpenMP is a build/runtime system dependency. Do not add other runtime deps.

## Python Version Compatibility

When adding support for a new Python version:

1. Update `requires-python` in `pyproject.toml` (if changing the floor).
2. Add the version classifier in `pyproject.toml`.
3. Add to the CI matrix in `.github/workflows/ci.yml` (both `test` and `build-wheels` jobs).
4. Add to the `cibuildwheel` `build = "..."` line in `pyproject.toml`.
5. Run the full test suite on the new version and verify the cimport test passes.

NumPy, SciPy, and Cython compatibility with the new Python version are the main risk factors.

## Key Rules

- **Do not refactor the numerical algorithms.** The WLSQM fit, the iterative refinement loop, the Ruiz 2001 scaling, and the polynomial evaluators are mathematically correct and performance-tested.
- **Do not rename public C-level API functions.** The `taylor_1D/2D/3D`, `interpolate_*`, `make_c_*`, `make_A`, `preprocess_A`, `solve`, and the LAPACK wrappers are all part of the Cython API that downstream projects `cimport`. Renaming them would break existing users. In particular, `taylor_*` stays `taylor_*` even though the comment block at the top of `polyeval.pyx` explains at length that it is not a Taylor series.
- **Do not remove OpenMP.** Parallel fitting across independent local problems is one of the headline features.
- **Do not convert `cdef struct` to `cdef class`.** The struct layout is what lets `Case_new` run from inside `nogil` parallel loops without touching the Python object allocator.
- **Do not change the Cython compiler directives** (`wraparound = False`, `boundscheck = False`, `cdivision = True`). They are intentional performance settings for numerical code.
- **Do not describe WLSQM as a "Taylor series" method.** The DOF layout looks like one, but the coefficients come from a least-squares fit — not analytic differentiation — and the error behavior is correspondingly much better than Taylor truncation would predict. `polyeval.pyx`'s header block has the full framing.
- **Do not convert the `defs.pxd` / `defs.pyx` constant system to `cdef enum` or Python `enum`.** The index and bitmask values are load-bearing — see "The `defs` constant system is NOT an enum" above. `ScalingAlgo` in `lapackdrivers.pyx` is the one exception; it is a genuine enum and IS now `IntEnum`.
- **Keep all six scaling algorithms in `lapackdrivers.pyx`.** They were compared experimentally; the library is a legitimate home for the comparison.

## Technical documentation

The `doc/` directory contains theory PDFs:

- `doc/wlsqm_gen.pdf` — the generalized version (including unknown function values), the accuracy analysis, and why WLSQM works.
- `doc/wlsqm.pdf` — older writeup for the pure-Python version originally written for FREYA, plus the sensitivity calculation.
- `doc/eulerflow.pdf` — application example in compressible flow, cleaner presentation of the original version.

A future documentation / tutorial pass is deferred — see `TODO_DEFERRED.md`.
