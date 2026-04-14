# wlsqm Source Audit Report

*Generated April 2026, prior to modernization. Companion to `wlsqm-modernization-brief.md` — the brief is the plan, this is what I verified before starting.*

---

## 1. Source inventory (verified)

| Module | Lines | Role | OpenMP | SciPy cimport |
|--------|------:|------|:------:|:-------------:|
| `wlsqm/fitter/defs.pyx`       |  504 | constants (Python + `*_c`) | no | no |
| `wlsqm/fitter/infra.pyx`      |  927 | Allocator / CaseManager / Case structs | no | no |
| `wlsqm/fitter/impl.pyx`       | 1090 | distance matrix, A assembly, solve | yes | no |
| `wlsqm/fitter/polyeval.pyx`   |  981 | Horner-form polynomial eval (fma) | no | no |
| `wlsqm/fitter/interp.pyx`     |  946 | model interpolation | no | no |
| `wlsqm/fitter/simple.pyx`     | 1170 | simple fit API | yes | no |
| `wlsqm/fitter/expert.pyx`     |  977 | ExpertSolver | yes | no |
| `wlsqm/utils/lapackdrivers.pyx` | 1745 | LAPACK wrappers + parallel | yes | **yes** |
| `wlsqm/utils/ptrwrap.pyx`     |   16 | `PointerWrapper` cdef class | no | no |
| **total** | **8356** | | | |

`.pxd` split mirrors the `.pyx` set (9 pxd files). Only one `cdef class` in the whole repo: `PointerWrapper`.

## 2. Build-order graph (confirmed by cimports)

Leaf → root: `defs`, `ptrwrap` → `infra`, `polyeval` → `lapackdrivers` → `interp` → `impl` → `simple`, `expert`.

`polyeval` does NOT cimport `infra`, contrary to what one might assume — it only needs `defs`. `interp` cimports `defs`, `polyeval`, `infra`.

`lapackdrivers.pyx` line 79: `from scipy.linalg.cython_lapack cimport dgeequ, dgesvd` — hence SciPy in `build-system.requires`.

## 3. Python 2 artifacts

`from __future__ import …` appears in **22 real source files** (plus the brief itself and `setup.py`):

- 9 `.pyx` files: `fitter/{defs,expert,impl,infra,interp,polyeval,simple}.pyx`, `utils/{lapackdrivers,ptrwrap}.pyx`
- 7 `.pxd` files: `fitter/{defs,impl,infra,interp,polyeval,simple}.pxd`, `utils/ptrwrap.pxd` (and `lapackdrivers.pxd`)
- `wlsqm/__init__.py`
- 4 example scripts under `examples/`
- `setup.py` (to be deleted)

Per `wlsqm-modernization-brief.md` §6, all `__future__` lines are deletable (Python ≥3.11 target).

No `DEF` / `IF` compile-time directives. No `cPickle`, `sympy.mpmath`, or `pkg_resources` (confirmed via grep — not present).

## 4. ScalingAlgo: confirmed ready for `IntEnum` conversion

`lapackdrivers.pyx:297`:
```cython
ctypedef int (*rescale_func_ptr)(double*, int, int, double*, double*) nogil
class ScalingAlgo:  # enum (TODO: use real enum type for Python 3.4+)
    ALGO_COLS_EUCL = 1
    ALGO_ROWS_EUCL = 2
    ALGO_TWOPASS   = 3
    ALGO_RUIZ2001  = 4
    ALGO_SCALGM    = 5
    ALGO_DGEEQU    = 6
```

Dispatcher `do_rescale` (line 306) compares `algo == ScalingAlgo.ALGO_*`. `IntEnum` members compare equal to their int values, so the switch keeps working. The author already flagged this with a TODO.

## 5. `rescale_dgeequ_c` bug confirmed

`lapackdrivers.pyx:498-504`:
```cython
# FIXME: Unfortunately, we must ignore exceptions to fit the rescale_func_ptr signature.
cdef int rescale_dgeequ_c( double* A, int nrows, int ncols, double* row_scale, double* col_scale ) nogil:
    cdef double rowcnd, colcnd, amax
    cdef int info
    dgeequ( &nrows, &ncols, A, &nrows, row_scale, col_scale, &rowcnd, &colcnd, &amax, &info )
    return 1
```

LAPACK's `dgeequ` sets `info > 0` when a row/column is exactly zero (singular) and `info < 0` for bad args. Both are silently dropped. The fix per brief §5.1 is:

1. Check `info` after the call.
2. Return 0 on failure (success → 1). This keeps the function pointer signature (`int (...) nogil`) so all other `rescale_*_c` functions need no signature change — they already return 1 unconditionally, so "0 = error, non-zero = ok" is a backward-compatible encoding.
3. Update `do_rescale` to inspect the return value after the `with nogil:` block and raise `RuntimeError`/`np.linalg.LinAlgError` on 0.

## 6. `fma` workaround in `polyeval.pyx` (lines 21-27)

```cython
# BUG in Cython 0.20.1post0: /usr/lib/python2.7/dist-packages/Cython/Includes/libc/math.pxd
# defines fma(double x, double y), but it should be fma(double x, double y, double z)
# so we import it manually.
#
#from libc.math cimport fma   # this works in newer Cythons, the bug has been fixed
cdef extern from "<math.h>" nogil:
    double fma(double x, double y, double z)
```

Replace the `cdef extern` block with `from libc.math cimport fma`. Delete the comment block. The author already left the correct line commented out.

## 7. `language_level` directive

Grep for `language_level` across `wlsqm/` returned **no matches**. The modules currently compile with Cython 0.29's default (implicit language level 2). Adding `add_project_arguments('-X', 'language_level=3', language: 'cython')` to the top-level `meson.build` is sufficient; no per-file directive removal needed.

## 8. Per-file Cython directives (keep)

All `.pyx`/`.pxd` files (except leaf utilities like `defs`, `ptrwrap`) carry:
```cython
# cython: wraparound  = False
# cython: boundscheck = False
# cython: cdivision   = True
```
These are intentional performance settings. Keep as-is per brief §5.4.

## 9. "Taylor series" misnomer sites

Files with the word "Taylor" (excluding `.html` artefacts and `defs.pyx` constant docstrings):

- `wlsqm/fitter/polyeval.pyx` — "Evaluate an up to 4th order Taylor series expansion in 3D space, with its origin at (xi,yi,zi)." plus similar 1D/2D/3D comments and the module-level docstring.
- `wlsqm/fitter/polyeval.pxd` — module comment.
- `wlsqm/fitter/interp.pyx` — docstrings referring to "Taylor coefficients."
- `wlsqm/fitter/impl.pyx` — docstring mentions.
- `wlsqm/fitter/defs.pyx` — the `i*_*` index constant docstrings. Here the word appears in the sense "index into the Taylor-style DOF vector"; reword to "polynomial coefficient index" but keep the constant names.

The C-level function names `taylor_1D` / `taylor_2D` / `taylor_3D` are part of the published Cython API and **stay**. Only docstrings and comments change.

## 10. Popcount portability — already handled

`wlsqm/fitter/popcount.h` exists and already aliases GCC's `__builtin_popcount` / `__builtin_popcountll` to MSVC's `__popcnt` / `__popcnt64` under `_MSC_VER`. No meson work needed on this front — just install `popcount.h` alongside `infra.pyx` so the Cython extension finds it during compilation. (This is subtler than installing a `.pxd`: it's a private header in the source tree, no install needed for wheels; only the source-dir include path during compilation matters.)

## 11. OpenMP usage sites (confirmed)

- `wlsqm/utils/lapackdrivers.pyx` — `cimport cython.parallel`, `cimport openmp`, multiple `prange` loops, `omp_get_thread_num()` calls (the `*p_c` parallel variants — `msymmetricp_c`, `mgeneralp_c`, etc.).
- `wlsqm/fitter/impl.pyx` — uses `cython.parallel.prange` in the multi-case helpers.
- `wlsqm/fitter/simple.pyx` — the `generic_fit_*_many_parallel` dispatchers.
- `wlsqm/fitter/expert.pyx` — `prange` inside `ExpertSolver.prepare()`/`solve()`.

Other extensions (`defs`, `infra`, `polyeval`, `interp`, `ptrwrap`) do NOT need the OpenMP dependency passed to them.

## 12. `ptrwrap.PointerWrapper.set_ptr`

`utils/ptrwrap.pyx:14`:
```cython
cdef class PointerWrapper:
    cdef set_ptr(self, void * input):
        self.ptr = input
```
GIL-holding, implicit object return. Under Cython 3 the implicit exception spec is `except *` for object-returning cdef methods; it's harmless but needlessly noisy. Either leave it or change to `cdef void set_ptr(self, void* input) noexcept:`. Low priority.

## 13. Version / metadata touches

- `wlsqm/__init__.py` line 24: `__version__ = '0.1.6'` → replace with `VERSION`-file read, same pattern as PyLU.
- `setup.py` author email: `juha.jeronen@jyu.fi` → `juha.jeronen@jamk.fi` (JAMK move). Grep for other `jyu.fi` hits during Phase 8.
- No `tests/` directory exists. No CI configuration exists.

---

## 14. Open questions / things to watch

- **popcount include path:** meson `py.extension_module(..., include_directories: include_directories('.'))` for `infra` so `popcount.h` is found at compile time. Verify this once the build works.
- **OpenMP graceful fallback:** Cython's `prange` already downgrades to `range` without `-fopenmp`, but `openmp.omp_get_thread_num()` produces an unresolved symbol. Needs empirical check; per brief §3.4 don't add a shim preemptively. If the first build on a libomp-less platform fails, add a tiny stub via `-DWLSQM_NO_OPENMP` + a Cython conditional. Defer until we see the failure.
- **Issue #5 (non-contiguous memoryview):** will revisit if it surfaces naturally while touching `expert.pyx` input handling.
- **`simple.pxd` / `defs.pxd` compiler directives as comments:** they are just plain comments in `.pxd` files (where Cython directive comments have no effect), so deleting the comment blocks from `.pxd` files is cosmetic only. Leave them.

This is enough to start Phase 2.
