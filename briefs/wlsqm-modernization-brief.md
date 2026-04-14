# python-wlsqm Modernization Brief

*For Claude Code. April 2026.*

## 1. Overview

python-wlsqm is a Cython+OpenMP library that constructs piecewise polynomial surrogate models via weighted least squares on scattered point clouds in 1D, 2D, and 3D. It was written in 2016–2017 for Python 2.7/3.4. The goal is to modernize it to run on Python ≥3.11, with modern build tooling, CI, and a PyPI release as v1.0.0.

**Repository:** `~/Documents/koodit/python-wlsqm`
**Current version:** 0.1.6 (2017)
**PyPI package name:** `wlsqm` (not `python-wlsqm`)
**License:** BSD-2-Clause

### 1.1 Dependencies

- **NumPy** — array operations, typed memoryviews
- **SciPy** — Cython-level LAPACK bindings (`scipy.linalg.cython_lapack`), used in `lapackdrivers.pyx`. **Both a build-time and runtime dependency** (the `cimport` must resolve at compile time).
- **OpenMP** — parallelization of independent local problems. See section 3.

No dependency on PyLU or pydgq. Those are sibling projects, not upstream. Sources for reference: PyLU at `~/Documents/koodit/pylu`, pydgq at `~/Documents/koodit/pydgq`.

### 1.2 What this project is NOT like PyLU or pydgq

Read this section carefully. Some patterns from those projects carry forward, but wlsqm has qualitatively different challenges.

1. **OpenMP parallelization.** Neither PyLU nor pydgq use OpenMP. wlsqm uses `cython.parallel.prange` and `openmp.omp_get_thread_num()` heavily in `lapackdrivers.pyx`, `expert.pyx`, and `simple.pyx`. The meson build must handle OpenMP compiler/linker flags across platforms (GCC, Clang, MSVC), with graceful fallback to serial on platforms without OpenMP. See section 3.

2. **SciPy as a Cython-level dependency.** pydgq depended on PyLU at Cython level; wlsqm depends on SciPy at Cython level (`from scipy.linalg.cython_lapack cimport dgtsv, dsysv, ...`). SciPy must be in `build-system.requires`.

3. **Split `noexcept` audit.** pydgq's audit was straightforward: almost everything got `noexcept`. In wlsqm, ~20 LAPACK wrapper functions in `lapackdrivers.pyx` use `nogil except -1` to propagate LAPACK errors as Python exceptions. These must KEEP `except -1`. Only the pure computational helpers (copy, symmetrize, rescale, min/max) and the functions in `infra`, `impl`, `polyeval`, `interp` should get `noexcept`. See section 5 for the full audit guide.

4. **C structs, not cdef classes.** pydgq had deep `cdef class` hierarchies requiring careful `noexcept` inheritance matching. wlsqm's hot-path data structures (`Allocator`, `BufferSizes`, `CaseManager`, `Case`) are all `cdef struct` in `infra.pxd`, with C-style constructor/destructor functions (`Case_new`, `Case_del`, etc.). Only one `cdef class` exists: `PointerWrapper` in `utils/ptrwrap.pyx` — a trivial utility with no inheritance. The cdef class report (`~/Documents/koodit/pydgq/briefs/cython3-cdef-class-report.md`) is useful background but applies only tangentially.

5. **No binary data files.** Unlike pydgq, there is no precalculated data file. The `pickle → npz` and `pkg_resources → importlib.resources` patterns do not apply.

6. **No DEF/IF directives.** Unlike pydgq, wlsqm has no `DEF` or compile-time `IF` directives to replace.

7. **No existing tests.** pydgq had test scripts to convert; wlsqm has zero tests. Tests must be written from scratch, informed by the examples in `examples/`. This is the largest new-work item in the modernization.

8. **Comment/docstring fix: "Taylor series" → surrogate model.** The existing code (especially `polyeval.pyx`) talks about "Taylor series expansion" and "Taylor coefficients." This is misleading — the polynomial has the same monomial basis as a Taylor expansion, but the coefficients come from a weighted least-squares fit, not from evaluating derivatives at a point. The error behavior is much better than Taylor truncation error would predict, because of the averaging effect of least squares. Fix these comments/docstrings during this pass: say "polynomial" or "local polynomial model," not "Taylor series." See section 8 for scope.

### 1.3 What IS like PyLU/pydgq (carries forward)

- meson-python + PDM + VERSION file (single source of truth)
- cibuildwheel + trusted publisher
- cython-lint
- `from __future__` removal (18 files)
- `.pxd` file installation (critical for downstream `cimport`)
- README / CHANGELOG / LICENSE / CLAUDE.md documentation pass
- Email update to JAMK (`juha.jeronen@jamk.fi`)
- Version → 1.0.0

### 1.4 Reference materials

- **Lessons learned:** `~/Documents/koodit/pydgq/briefs/modernization-lessons-learned.md`
- **Cython 3 cdef class report:** `~/Documents/koodit/pydgq/briefs/cython3-cdef-class-report.md`
- **PyLU build reference:** `~/Documents/koodit/pylu/pyproject.toml`, `~/Documents/koodit/pylu/meson.build`, `~/Documents/koodit/pylu/.github/workflows/ci.yml`
- **pydgq brief (structural template):** `~/Documents/koodit/pydgq/briefs/pydgq-modernization-brief.md`
- **Mathematical theory:** `doc/wlsqm_gen.pdf` in this repo (general theory, accuracy analysis)

---

## 2. Source Inventory

### 2.1 Package structure

```
wlsqm/
├── __init__.py           # version, re-exports from fitter submodules
├── fitter/
│   ├── __init__.py       # empty
│   ├── defs.pxd          # C-level constant declarations (231 lines)
│   ├── defs.pyx          # constant values + Python-accessible names (504 lines)
│   ├── infra.pxd         # C structs: Allocator, BufferSizes, CaseManager, Case (196 lines)
│   ├── infra.pyx         # struct constructors/destructors, memory management (927 lines)
│   ├── impl.pxd          # low-level routine declarations (49 lines)
│   ├── impl.pyx          # core fitting implementation (1090 lines)
│   ├── polyeval.pxd      # polynomial evaluation declarations (23 lines)
│   ├── polyeval.pyx      # Horner-form polynomial evaluation with FMA (981 lines)
│   ├── interp.pxd        # interpolation declarations (22 lines)
│   ├── interp.pyx        # model interpolation at arbitrary points (946 lines)
│   ├── simple.pxd        # "driver" API declarations (68 lines)
│   ├── simple.pyx        # simple "driver" API (1170 lines)
│   ├── expert.pyx        # ExpertSolver Python class + advanced API (977 lines)
│   └── (no expert.pxd)
├── utils/
│   ├── __init__.py       # empty
│   ├── lapackdrivers.pxd # LAPACK wrapper declarations (131 lines)
│   ├── lapackdrivers.pyx # LAPACK wrappers + OpenMP parallel solvers (1745 lines)
│   ├── ptrwrap.pxd       # PointerWrapper cdef class declaration (14 lines)
│   └── ptrwrap.pyx       # PointerWrapper implementation (16 lines)
```

Total: ~9100 lines of Cython, across 9 extensions + 10 pxd/pyx declaration files.

### 2.2 Inter-module cimport graph

| Module | cimports |
|--------|----------|
| `defs.pyx` | (none — leaf) |
| `infra.pyx` | `defs` |
| `polyeval.pyx` | `defs` |
| `lapackdrivers.pyx` | `openmp`, `cython.parallel`, `scipy.linalg.cython_lapack` |
| `impl.pyx` | `defs`, `infra`, `interp`, `lapackdrivers` |
| `interp.pyx` | `defs`, `polyeval`, `infra` (via pxd) |
| `expert.pyx` | `openmp`, `cython.parallel`, `ptrwrap`, `defs`, `infra`, `impl`, `interp` |
| `simple.pyx` | `openmp`, `cython.parallel`, `defs`, `infra`, `impl` |
| `ptrwrap.pyx` | (none — leaf) |

**Build order (leaf → root):** `defs`, `ptrwrap` → `infra`, `polyeval` → `lapackdrivers` → `interp` → `impl` → `simple`, `expert`

### 2.3 Extension modules to build

| Extension | Package | Math/OpenMP | External Cython deps |
|-----------|---------|-------------|---------------------|
| `defs` | `wlsqm.fitter` | no | — |
| `infra` | `wlsqm.fitter` | yes (math) | — |
| `impl` | `wlsqm.fitter` | yes (math+OpenMP) | — |
| `polyeval` | `wlsqm.fitter` | yes (math, fma) | — |
| `interp` | `wlsqm.fitter` | yes (math) | — |
| `simple` | `wlsqm.fitter` | yes (math+OpenMP) | — |
| `expert` | `wlsqm.fitter` | yes (math+OpenMP) | — |
| `lapackdrivers` | `wlsqm.utils` | yes (math+OpenMP) | `scipy.linalg.cython_lapack` |
| `ptrwrap` | `wlsqm.utils` | no | — |

---

## 3. OpenMP Handling

This is the main new challenge relative to PyLU/pydgq.

### 3.1 Which files use OpenMP

- `wlsqm/utils/lapackdrivers.pyx` — 10+ `prange` loops, `omp_get_thread_num()`
- `wlsqm/fitter/expert.pyx` — 15+ `prange` loops, `omp_get_thread_num()`
- `wlsqm/fitter/simple.pyx` — `prange` in the `*_many_parallel` functions

All use `cimport cython.parallel` and `cimport openmp`.

### 3.2 Meson build: OpenMP detection

Use meson's built-in OpenMP dependency:

```meson
omp_dep = dependency('openmp', required: false)
if omp_dep.found()
    message('OpenMP found — building with parallel support')
else
    message('OpenMP NOT found — building serial-only (reduced performance)')
endif
```

Pass `omp_dep` to all extension modules that use OpenMP (lapackdrivers, impl, simple, expert). Extensions that don't use OpenMP (defs, infra, polyeval, interp, ptrwrap) do not need it.

### 3.3 Platform notes

- **Linux (GCC):** `-fopenmp` for compile and link. Works out of the box.
- **macOS (Apple Clang):** Does NOT have OpenMP by default. `libomp` can be installed via Homebrew (`brew install libomp`), but we cannot assume it's present. The `required: false` fallback handles this — users get a serial build, with a note in the README explaining how to enable OpenMP on macOS.
- **Windows (MSVC):** `/openmp` flag. Meson handles this automatically via the `openmp` dependency.
- **Windows (MinGW):** `-fopenmp` as on Linux. Meson handles this.

### 3.4 Graceful degradation

When OpenMP is not available, Cython's `prange` falls back to a regular `range` (this is built into Cython — no code changes needed). The `cimport openmp` and `openmp.omp_get_thread_num()` calls do need guarding. Check whether Cython handles this automatically when compiled without OpenMP, or whether a compile-time flag is needed.

If `omp_get_thread_num()` causes a compile error without OpenMP, the simplest fix is a small C shim:

```meson
# In meson.build, for modules that use omp_get_thread_num:
if not omp_dep.found()
    add_project_arguments('-DWLSQM_NO_OPENMP', language: 'c')
endif
```

And in the Cython code, a helper that returns 0 when OpenMP is disabled. **However**: investigate first whether this is actually needed. Cython may handle it. Don't add complexity preemptively.

### 3.5 README note

Add to the README: "On platforms without OpenMP (notably macOS with Apple Clang), wlsqm builds in serial mode. All functionality works, but parallel features (`ntasks` parameter) are disabled and performance will be significantly lower. For parallel support on macOS, install `libomp` via Homebrew (`brew install libomp`) and compile from source:

```bash
brew install libomp
pip install --no-binary wlsqm wlsqm
```

Meson should automatically detect `libomp` once it is installed. If it doesn't, set `CFLAGS` and `LDFLAGS` to point at the Homebrew `libomp` location."

### 3.6 CI strategy

See section 7.3 for macOS `libomp` installation in CI wheel builds, and section 7.4 for serial fallback testing.

---

## 4. Build System Migration

### 4.1 From setup.py to meson-python + PDM

Same pattern as PyLU/pydgq:

- `pyproject.toml` — package metadata, build system, tool config, PDM dev deps
- `meson.build` (top-level) — project declaration, OpenMP detection, subdir calls
- `wlsqm/meson.build` — `__init__.py`, subdir calls
- `wlsqm/fitter/meson.build` — all fitter `.pyx` extensions, `.pxd` installs
- `wlsqm/utils/meson.build` — utils `.pyx` extensions, `.pxd` installs

Dev workflow:
```bash
pdm config venv.in_project true
pdm use 3.14                             # or whichever version is current
pdm install                              # creates .venv, installs dev deps
export PATH="$(pwd)/.venv/bin:$PATH"     # meson/ninja on PATH for editable rebuilds
pip install --no-build-isolation -e .     # editable install via meson-python
```

The `--no-build-isolation` flag is required for editable installs with meson-python — the on-import rebuild mechanism needs build dependencies to remain available in the environment. The PATH export is needed so that meson and ninja (installed by PDM into the venv) are found during on-import rebuilds.

See `~/Documents/koodit/pylu/` for the reference implementation of this setup, and `~/.claude/PROJECT-SETUP-NOTES.md` for the canonical Cython project setup recipe.

### 4.2 Version: single source of truth

```
wlsqm/VERSION  — plain text, e.g. "1.0.0"
meson.build    — version: files('wlsqm/VERSION')
pyproject.toml — dynamic = ["version"]
__init__.py    — from pathlib import Path as _Path
                 __version__ = (_Path(__file__).parent / "VERSION").read_text().strip()
```

### 4.3 Compiler flags

The current `setup.py` uses `-march=native -O2 -msse -msse2 -mfma -mfpmath=sse` for math-heavy extensions. These are NOT suitable for wheels (non-portable).

**For wheels (CI builds):** Use meson's `buildtype=release` (gives `-O2`). No arch-specific flags. OpenMP flags are handled by the `openmp` dependency object.

**For source builds:** Document in README how to enable architecture-specific optimizations via `CFLAGS`:
```bash
CFLAGS="-march=native" pip install --no-binary wlsqm wlsqm
```
This enables all instruction sets available on the build machine (SSE, AVX, FMA, etc.). The old `setup.py` listed `-msse -msse2 -mfma -mfpmath=sse` explicitly, but these are all implied by `-march=native` on x86_64 (SSE/SSE2 are part of the x86_64 baseline; FMA is enabled if the CPU has it). No need to list them separately.

**libm:** Handled by meson's `cc.find_library('m', required: false)`. Pass to math-heavy extensions.

**`_USE_MATH_DEFINES`:** Not strictly needed (wlsqm doesn't use `M_PI`), but add it defensively — it's a no-op on Linux/macOS and costs nothing:
```meson
add_project_arguments('-D_USE_MATH_DEFINES', language: 'c')
```

### 4.4 pyproject.toml

```toml
[build-system]
requires = ["meson-python>=0.17", "Cython>=3.0", "numpy>=1.25", "scipy>=1.9"]
build-backend = "mesonpy"

[project]
name = "wlsqm"
dynamic = ["version"]
description = "Fast and accurate weighted least squares meshless interpolator for Python"
readme = "README.md"
license = "BSD-2-Clause"
requires-python = ">=3.11"
authors = [{name = "Juha Jeronen", email = "juha.jeronen@jamk.fi"}]
keywords = [
    "numerical", "interpolation", "differentiation",
    "curve-fitting", "least-squares", "meshless",
    "numpy", "cython",
]
dependencies = [
    "numpy>=1.25",
    "scipy>=1.9",
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Programming Language :: Python :: 3.14",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
]

[project.urls]
Homepage = "https://github.com/Technologicat/python-wlsqm"
Repository = "https://github.com/Technologicat/python-wlsqm"
Issues = "https://github.com/Technologicat/python-wlsqm/issues"
Changelog = "https://github.com/Technologicat/python-wlsqm/blob/master/CHANGELOG.md"

[tool.ruff]
line-length = 130
target-version = "py311"
exclude = [
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".venv",
    "00_stuff",
]

[tool.ruff.lint]
select = ["E", "W", "F", "SIM"]
ignore = [
    # pycodestyle
    "E203",   # whitespace before ':' — needed for slice alignment
    "E265",   # block comment should start with '# ' — commented-out code, markers
    "E301",   # expected 1 blank line — blank lines are semantic paragraph breaks
    "E302",   # expected 2 blank lines before def — same
    "E305",   # expected 2 blank lines after end — same
    "E306",   # expected blank line before nested def — same
    "E402",   # module level import not at top — conditional/deferred imports
    "E501",   # line too long — advisory, not enforced
    "E731",   # lambda assignment — closures are idiomatic in this codebase
    # flake8-simplify
    "SIM102",  # collapsible if — nested ifs often represent distinct semantic guards
    # Note: SIM103 (return condition directly) is intentionally NOT ignored here.
    # It is enabled as an advisory — CI runs it in a non-failing second pass.
    "SIM105",  # contextlib.suppress — try/except/pass is more flexible and explicit
    "SIM108",  # ternary instead of if/else — often less readable, no real gain
    "SIM114",  # combine if branches — match-casing style; autofix would damage semantics
    "SIM117",  # combine with statements — nesting shows parent/child relationships
    "SIM118",  # in-dict-keys — explicit .keys() marks the variable as a dictlike
    "SIM300",  # yoda conditions — natural reading order preferred
    "SIM910",  # dict.get with None default — explicit None documents programmer intent
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401", "F403"]  # re-exports via star-import

[tool.cython-lint]
max-line-length = 200
ignore = ["E115", "E201", "E202", "E221", "E231", "E302", "W293", "W391"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pip",
    "cython-lint",
    "Cython>=3.0",
    "ruff>=0.14.0",
    "flake8",
    "autopep8",
    "importmagic",
    "epc",
    "jedi>=0.19.2",
    "scipy>=1.9",
    "matplotlib",
    # Needed in the venv (not just the isolated build env) so meson-python's
    # editable loader can rebuild the extension on import after .pyx/.pxd edits.
    "meson-python>=0.17",
    "meson",
    "ninja",
]
```

Note: `scipy` is in both `build-system.requires` (for `cimport` during compilation) and `project.dependencies` (runtime). It's also in `dev` deps for the editable install to work (due to `--no-build-isolation`).

---

## 5. Cython 3.x Migration

### 5.1 `noexcept` audit

This is the critical-path item. wlsqm has a split personality:

**Functions that SHOULD get `noexcept`** — pure computational helpers that never raise:

In `lapackdrivers.pyx` / `.pxd`:
- `cimin`, `cimax` (inline helpers)
- `distribute_items_c`
- `copygeneral_c`, `copysymmu_c`
- `symmetrize_c`, `msymmetrize_c`, `msymmetrizep_c`
- `init_scaling_c`, `apply_scaling_c`
- `basic_scale_up_rows`, `basic_scale_up_cols`, `basic_scale_down_rows`, `basic_scale_down_cols`

In `infra.pxd` / `.pyx`:
- `number_of_dofs`, `number_of_reduced_dofs`, `remap`
- `Allocator_malloc`, `Allocator_free`, `Allocator_size_remaining`, `Allocator_del`
- `CaseManager_del`
- `Case_get_wrk`, `Case_get_fk_tmp`, `Case_get_fi_tmp`
- `Case_make_weights`, `Case_set_fi`, `Case_get_fi`, `Case_del`

In `impl.pxd` / `.pyx`:
- `make_c_nD`, `make_c_3D`, `make_c_2D`, `make_c_1D`
- `make_A`, `preprocess_A`
- `solve`

In `polyeval.pxd` / `.pyx`:
- All `taylor_*` and `general_*` functions

In `interp.pxd` / `.pyx`:
- All `interpolate_*` functions

In `ptrwrap.pxd` / `.pyx`:
- `PointerWrapper.set_ptr`

**Functions that MUST KEEP `except -1`** — LAPACK wrappers that propagate errors:

In `lapackdrivers.pyx` / `.pxd`:
- ALL `symmetric*_c`, `general*_c`, `msymmetric*_c`, `mgeneral*_c` functions
- `svd_c`
- `tridiag` (cpdef)

In `simple.pxd` / `.pyx`:
- ALL `generic_fit_*` functions (they call LAPACK wrappers internally)

In `infra.pxd` / `.pyx`:
- `Allocator_new` (uses `except <Allocator*>0`)
- `CaseManager_new` (uses `except <CaseManager*>0`)
- `CaseManager_add`, `CaseManager_commit` (use `except -1`)
- `Case_new` (uses `except <Case*>0`)

In `lapackdrivers.pxd`:
- The `rescale_*_c` functions return `int` and are called through a function pointer typedef (`rescale_func_ptr`, line 296) declared as `nogil` without `except`. The hand-crafted ones (columns, rows, twopass, ruiz2001, scalgm) are pure computation and should get `noexcept`. **Exception: `rescale_dgeequ_c`** — see known bug below.

**Known bug in `rescale_dgeequ_c`:** This function (line 499) calls SciPy's `dgeequ` LAPACK binding and **ignores the `info` return value**, unconditionally returning 1. LAPACK's `dgeequ` sets `info > 0` when a row or column is exactly zero (indicating a singular or near-singular matrix), and `info < 0` for argument errors. This should be fixed during this pass. The fix involves:
1. Checking `info` after the `dgeequ` call
2. Propagating the error — but note the function pointer typedef `rescale_func_ptr` at line 296 is `ctypedef int (*rescale_func_ptr)(...) nogil` without exception spec, so exception propagation through the pointer is not possible. The return value (`int`) should be used for error signaling instead (e.g., return 0 on error, 1 on success), with the caller (`do_rescale` at line 306) checking the return value after the `with nogil:` block.
3. All other `rescale_*_c` functions already return 1 unconditionally, so this change is backward-compatible with the function pointer interface — just check the return value in `do_rescale`.

**Audit technique:**
1. `grep -n 'cdef.*nogil' *.pyx *.pxd` in each subpackage
2. For each hit, determine: does it call anything that can raise? If no → `noexcept`. If yes → keep `except -1` or existing exception spec.
3. Verify `.pxd` and `.pyx` signatures match exactly after changes.
4. Run the audit twice — once to make changes, once to verify.

### 5.2 Language level

Add globally in top-level `meson.build`:

```meson
add_project_arguments('-X', 'language_level=3', language: 'cython')
```

This replaces per-file `# cython: language_level=3` directives and avoids missing a file. Drop the per-file directives after adding this.

### 5.3 fma import fix

In `polyeval.pyx`, replace the old workaround:
```cython
# Old (worked around Cython 0.20.1 bug):
cdef extern from "<math.h>" nogil:
    double fma(double x, double y, double z)

# New:
from libc.math cimport fma
```

The bug was fixed long ago; `libc.math cimport fma` works correctly in Cython ≥3.0.

### 5.4 Cython compiler directives

The existing per-file directives are fine:
```cython
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
```

These can stay as per-file comments, or be moved to the global meson config. Per-file is fine and more explicit. Do NOT change their values — they are intentional performance settings for numerical code.

---

## 6. Python 2 Removal

Mechanical changes across all `.py` and `.pyx` files:

- Remove all `from __future__ import division, print_function, absolute_import` (18 files)
- Remove `from __future__ import absolute_import` from `.pxd` files (`defs.pxd`, `simple.pxd`)
- Remove the Python 2.7 version check in `setup.py` (setup.py itself is being deleted)
- Remove Python 2 classifiers from metadata
- Update the `fma` workaround comment referencing Cython 0.20.1 (remove the comment block; the fix in 5.3 makes it irrelevant)
- **Convert `ScalingAlgo`** in `lapackdrivers.pyx` (line 297) from a bare class to a proper `enum.IntEnum`. The current implementation is a Python 2.7 workaround (Python 2 had no `enum`). The `IntEnum` values work transparently as `int` in the function pointer dispatch in `do_rescale`, so the change is backward-compatible.

---

## 7. CI and Publishing

### 7.1 Follow the PyLU pattern

See `~/Documents/koodit/pylu/.github/workflows/ci.yml` for the reference implementation.

- **Lint job:** Two-pass ruff (`ruff check . --ignore SIM103` as blocking, `ruff check . --select SIM103 || true` as advisory non-blocking), cython-lint on all `.pyx` / `.pxd` files (non-blocking). See PyLU CI for the exact pattern.
- **Test job:** Python 3.11–3.14 × Linux/macOS/Windows. The CI installs pytest via `pip install pytest` (no `[test]` extra — pytest is a dev dependency, not a published optional extra).
- **Build wheels:** cibuildwheel ≥3.4
- **Publish job:** trusted publisher via GitHub, triggers on `v*` tags

### 7.2 cibuildwheel config

```toml
[tool.cibuildwheel]
build = "cp311-* cp312-* cp313-* cp314-*"
test-command = "pytest {project}/tests -v"
test-requires = ["pytest", "numpy", "scipy"]
```

### 7.3 OpenMP in CI

- **Linux:** OpenMP works by default with GCC.
- **macOS:** Install `libomp` before building wheels so the macOS wheels ship with OpenMP support:
  ```yaml
  - name: Install libomp (macOS)
    if: runner.os == 'macOS'
    run: brew install libomp
  ```
  Meson's `dependency('openmp')` should find it automatically. If not, set `CFLAGS`/`LDFLAGS` in the cibuildwheel environment to point at Homebrew's `libomp` location (`/opt/homebrew/opt/libomp` on ARM, `/usr/local/opt/libomp` on Intel). The `delocate` wheel repair tool (used by cibuildwheel on macOS) should bundle `libomp.dylib` into the wheel.
- **Windows:** MSVC has built-in OpenMP support.

### 7.4 Serial fallback testing

Not strictly necessary in CI (no one would intentionally build numerics code without OpenMP where it's available), but if we want to be thorough, one macOS job without `libomp` installed would verify the `required: false` fallback compiles and runs. Low priority — defer unless easy to add.

### 7.5 Trusted publisher

Set up on PyPI for `wlsqm` package (note: package name is `wlsqm`, not `python-wlsqm`), workflow `ci.yml`, environment `pypi`. Same procedure as PyLU — see `~/.claude/CI-SETUP-NOTES.md`.

---

## 8. Comment and Docstring Fixes

### 8.1 "Taylor series" → polynomial / local polynomial model

The existing code refers to "Taylor series expansion", "Taylor coefficients", etc. in comments and docstrings. Fix these to accurately describe what the code does:

- "Taylor series expansion" → "local polynomial model" or "polynomial expansion"
- "Taylor coefficients" → "polynomial coefficients" or "model coefficients"
- The function names `taylor_1D`, `taylor_2D`, `taylor_3D` in `polyeval.pyx` — these are internal `cdef` functions and part of the C-level API, so **do NOT rename them** (would break any downstream `cimport` users). But update their docstrings.
- The `general_1D`, `general_2D`, `general_3D` functions are fine as-is.

**Scope:** Fix comments and docstrings in `polyeval.pyx`, `interp.pyx`, and any other files that use "Taylor" in a misleading way. Do NOT rename functions or change the API. Do NOT change the `defs.pxd` constant names.

### 8.2 WLSQM mathematical framing (for CLAUDE.md and README)

WLSQM constructs a piecewise polynomial surrogate model by weighted least squares. The polynomial has the same monomial basis as a Taylor expansion, but is NOT a Taylor series — it is a least-squares fit over a local neighborhood of data points. Do not describe it as a "Taylor approximation" in any new text. The error behavior is much better than Taylor truncation error would predict, due to the averaging effect of the least-squares fit. For full theory, see `doc/wlsqm_gen.pdf`.

The method is known under several names in the literature: MLS (moving least squares), WLSQM (weighted least squares meshless), "diffuse approximation." These are essentially the same idea. Some variants of MLS come with an explicit weighting function on the neighbor points (e.g. distance-based decay), which can stabilize the output against outliers. Adding weighting function support to wlsqm would be a natural extension — deferred (see post-session reminders).

---

## 9. Test Creation

### 9.1 Strategy

There are no existing tests. Create a `tests/` directory with proper pytest tests.

The `examples/` directory provides starting points:
- `expertsolver_example.py` — minimal ExpertSolver usage
- `wlsqm_example.py` — comprehensive tour of the API
- `lapackdrivers_example.py` — LAPACK wrapper usage
- `sudoku_lhs.py` — Latin hypercube sampling utility (keep in examples, not a test)

### 9.2 What to test

- **Import test:** `import wlsqm` succeeds, `wlsqm.__version__` is correct
- **cimport test:** verify `.pxd` files are installed (same pattern as PyLU)
- **Simple API (1D, 2D, 3D):** Fit a known polynomial, verify coefficients match within tolerance. Verify computed derivatives match analytical values.
- **Expert API:** Same as simple, but using ExpertSolver. Test prepare/solve separation.
- **Interpolation:** Fit a model, evaluate at known points, verify accuracy.
- **Multi-case parallel:** Fit many models at once with `ntasks > 1`, verify same results as serial.
- **LAPACK drivers:** Basic solve test (symmetric, general, tridiagonal). Test parallel variants.
- **Edge cases:** 0th order fit, 4th order fit, single neighbor point, minimum number of neighbors.
- **Regression for issue #5:** Test that passing non-contiguous arrays produces a clear error message (or works correctly, if it turns out to be a bug we can fix).

### 9.3 Test data generation

For verifying derivative computation, use known analytical functions. For example, in 2D: f(x,y) = x³ + 2x²y + 3xy² + y³. The derivatives are known exactly, so the fitted model's coefficients (with a fine enough point cloud and order ≥ 3) should match within numerical tolerance.

For stress-testing, generate random point clouds with `numpy.random` (seeded with `seed=42` — standard choice across all projects).

### 9.4 Multi-case parallel note

When testing multi-case parallel (`ntasks > 1`), optionally also verify that wall time is lower than the serial run — this confirms multiple CPUs actually got used. This is probably not reliable in CI (variable load on shared runners), but is valuable for local testing. Guard the timing assertion with a generous margin (e.g. parallel must be at least 1.5× faster than serial for ≥1000 cases) or mark it `@pytest.mark.skipif` in CI.

---

## 10. Documentation Updates

- **README.md:** Rewrite installation section (pip install, no more setup.py). Add badges (CI, PyPI, license). Add performance build tip (`CFLAGS="-march=native"`). Add OpenMP note for macOS (including source build instructions). Add semver policy and AI contributions link as separate paragraphs (one blank line between), placed after the project description and before the first section header, matching PyLU/pydgq:
  ```
  We use [semantic versioning](https://semver.org/).

  For my stance on AI contributions, see the [collaboration guidelines](https://github.com/Technologicat/substrate-independent/blob/main/collaboration.md).
  ```
- **CHANGELOG.md:** Add v1.0.0 entry.
- **LICENSE.md:** Update year range and affiliation to JAMK.
- **CLAUDE.md:** Create, following the structure of PyLU's (`~/Documents/koodit/pylu/CLAUDE.md`). Should include:
  - **What is wlsqm** — one-paragraph description, the surrogate-model-not-Taylor-series framing (section 8.2), dependency philosophy
  - **Build and Development** — PDM + meson-python workflow, editable install caveat (`--no-build-isolation`, PATH trick), VERSION file, OpenMP notes
  - **Running Tests** — `pdm run pytest tests/ -v`
  - **Architecture** — the C struct approach (not cdef classes), the `.pxd`/`.pyx` split, the `defs` constant system (see below), the simple vs expert API distinction, the `lapackdrivers` LAPACK wrapper layer, the OpenMP parallelization pattern
  - **The `defs` constant system** — `defs.pxd` declares `cdef int` module-level variables; `defs.pyx` assigns their values and also exports Python-accessible copies. This is NOT an enum pattern, even though `ALGO_*` and `WEIGHT_*` look like enums. The `i1_*`, `i2_*`, `i3_*` constants are array indices with specific numerical values that the fitting code depends on. The `knowns` parameter uses bitmasks. Do NOT convert these to `cdef enum` or Python `enum.Enum` — the values are load-bearing and the pattern works as-is. (The `ScalingAlgo` class in `lapackdrivers.pyx` is different — that one IS a classic enum and SHOULD be converted to `enum.IntEnum`; see section 6.)
  - **Critical constraint: .pxd installation** — all `.pxd` files must be installed for downstream `cimport`
  - **Linting** — ruff for `.py` files (CI uses two-pass: `ruff check . --ignore SIM103` blocking, `ruff check . --select SIM103 || true` advisory), cython-lint for `.pyx`/`.pxd` (non-blocking in CI), flake8 retained for Emacs IDE integration (global config at `~/.config/flake8`, not duplicated per-project). See `~/.claude/PROJECT-SETUP-NOTES.md` for canonical lint config.
  - **Code Conventions** — line width, docstring format, dependency policy
  - **Key Rules** — do not refactor numerical algorithms, do not rename C-level API functions, do not remove OpenMP, do not convert C structs to cdef classes, do not change compiler directives (wraparound, boundscheck, cdivision), do not describe WLSQM as a "Taylor series" method, do not convert the `defs` constant system to `cdef enum` or Python `enum` (the index and bitmask values are load-bearing), keep ALL scaling/preconditioning algorithms in `lapackdrivers.pyx` (there are several because they were compared experimentally; the library is a good home for them)
  - **Technical documentation** — note that `doc/` contains theory PDFs (`wlsqm.pdf`, `wlsqm_gen.pdf`), plus `eulerflow.pdf` (application example). Content update deferred to a future tutorial pass.

---

## 11. Existing Issues

GitHub has at least one open issue:

- **#5** `ValueError: Buffer and memoryview are not contiguous in the same dimension` — **Defer** to a separate pass. If the answer naturally comes up during the modernization (e.g. while auditing `expert.pyx`'s input handling), fix it then. Otherwise, note the finding in `TODO_DEFERRED.md` and move on.

Other issues (if any) should be triaged similarly: fix if the answer arises naturally during the migration, otherwise defer and document.

---

## 12. Migration Sequence

1. **Read all source files first.** Understand the module dependency graph, the OpenMP usage pattern, and the C struct APIs before touching anything. Write a source audit report (template: `~/Documents/koodit/pydgq/briefs/source-audit-report.md`).
2. **Create pyproject.toml and meson.build files.** Get the build working with meson-python, including OpenMP detection. Set up PDM dev workflow.
3. **Remove Python 2 artifacts.** `__future__` imports, old comments, version check. Convert `ScalingAlgo` to `enum.IntEnum`.
4. **Cython 3.x fixes.** Language level (global), `noexcept` audit (the split-personality audit — section 5.1 is your guide), `fma` import fix, `rescale_dgeequ_c` bug fix (section 5.1).
5. **Comment/docstring fixes.** "Taylor series" → "polynomial" / "local polynomial model" where misleading. Do NOT rename functions.
6. **Tests.** Write from scratch (section 9).
7. **CI setup.** Lint, test matrix, cibuildwheel, OpenMP CI handling, trusted publisher.
8. **Documentation.** README, CHANGELOG, LICENSE, CLAUDE.md.
9. **Verify.** All tests pass, cython-lint clean, CI green on all platforms.
10. **Tag and publish.** v1.0.0, push to PyPI via trusted publisher.

Note: if issue #5 (buffer/memoryview contiguity) naturally resolves itself during the migration (e.g. the answer becomes obvious while reading `expert.pyx`), fix it. Otherwise defer to `TODO_DEFERRED.md`.

---

## 13. Post-Session Reminders

- Bump version to `1.0.1.dev0` in VERSION file after release (note: PEP 440 format, not `1.0.1-dev`)
- Set up Dependabot (copy config from PyLU)
- Push the brief to `briefs/` in the repo
- Update `~/.claude/CLAUDE.md` project list to include wlsqm
- Check `doc/` for any source files that need noting in CLAUDE.md
- Future: documentation/tutorial pass for `doc/` content (deferred — potential arXiv tutorial target, surrogate model framing; method goes under many names: MLS, WLSQM, "moving least squares," "diffuse approximation")
- Future: add weighting function support (e.g. distance-based weights per neighbor) to stabilize output against outliers — a natural extension of the existing algorithm, and the main thing distinguishing different MLS variants in the literature
- Future: investigate `sudoku_lhs.py` extraction to separate library (deferred)
