# Deferred TODO

Notes on unrelated items noticed during the v1.0.0 modernization pass. Not blocking v1.0.0; revisit later.

## Matrix scaling: robustness parity with SciPy

- Our hand-rolled scalers in `wlsqm/utils/lapackdrivers.pyx`
  (`rescale_columns_c`, `rescale_rows_c`, `rescale_twopass_c`,
  `rescale_ruiz2001_c`, `rescale_scalgm_c`) do not detect singular rows/columns.
  LAPACK's `DGEEQU` does, and `rescale_dgeequ_c` now propagates that as
  `numpy.linalg.LinAlgError`. The others silently divide by the row/column
  norm without checking for zero, which for a pathological input will produce
  `inf`/`nan` scaling factors and poison the downstream solve. Add explicit
  zero-row/zero-column checks to each scaler and propagate via the existing
  `int` return-code channel.
- Also add SciPy-style input validation at the Python entry points
  (`rescale_*`, `do_rescale`): reject non-finite entries, non-square where
  required, zero-size arrays, wrong dtype/layout. Right now we rely on the
  Cython memoryview machinery to reject the most egregious cases, but the
  error messages are opaque.
- While at it, revisit `wlsqm.fitter.simple.fit_*` / `wlsqm.fitter.expert.ExpertSolver`
  input validation too — same flavour.

## Issue #5: non-contiguous memoryview ValueError

Open on GitHub: `ValueError: Buffer and memoryview are not contiguous in the
same dimension`. Reproduce, then decide: clearer error at the Python entry
point, or fix to accept non-contiguous inputs where cheap. Deferred from the
main modernization pass.

## `sudoku_lhs.py` extraction

`examples/sudoku_lhs.py` is a small Latin-hypercube helper that could stand
alone as a tiny package. Not urgent.

## Documentation / tutorial pass for `doc/`

`doc/wlsqm.pdf` and `doc/wlsqm_gen.pdf` are the original theory documents.
Potential arXiv tutorial target — rewrite them with the "surrogate model,
not Taylor series" framing, and mention that the same method appears in the
literature under several names (MLS, WLSQM, "diffuse approximation").

## Windows `long` width in ExpertSolver.interpolate

`ExpertSolver.interpolate()` in `wlsqm/fitter/expert.pyx` backs its
`I_out` return array with a Cython `long[::1]` view and an allocation of
`dtype=np.int_`. On Linux/macOS 64-bit, C `long` is 64 bits; on Windows
64-bit (MSVC) it is 32 bits. The function therefore silently produces
different element widths across platforms, and cannot address more than
2**31 local models on Windows. Low priority (no one has a WLSQM setup
with > 2 billion neighborhoods) but the fix is small: switch both the
cdef view type and the numpy dtype to a fixed width, e.g. `np.int64_t`
/ `np.int64`.

## Weighting function support

The literature distinguishes MLS variants mainly by the weighting function
applied to neighbor points (e.g. distance-based decay). wlsqm currently has
only `WEIGHT_UNIFORM` and `WEIGHT_CENTER`. Adding a user-supplied radial
weight function would be a natural extension and improves robustness against
outliers.

## Python 3.15 support, once SciPy ships `cp315` wheels

wlsqm is the one project in the fleet that cannot take 3.15 yet, and the block
is entirely external. `utils/lapackdrivers.pyx` does
`from scipy.linalg.cython_lapack cimport ...`, which makes SciPy a *build*
dependency, not merely a runtime one — so no SciPy on 3.15 means the extensions
do not compile at all, and neither the `test` job nor `cibuildwheel` can run.

Checked 2026-08-18 against PyPI: `pip install --only-binary=:all: scipy` under
3.15.0rc1 reports "no matching distribution", and no SciPy release publishes a
`cp315` wheel. Building SciPy from source inside a manylinux container to get
around this is not worth it — it needs a Fortran toolchain and BLAS, and the
whole point of the wheel job is that it is reproducible.

Expect this to clear itself some weeks after 3.15 final, when SciPy cuts a
release with `cp315` wheels. The check is one command; when it passes, the edit
is the same one the other two Cython projects already took:

```bash
pip install --only-binary=:all: --dry-run scipy   # under a 3.15 interpreter
```

- `pyproject.toml`: add `"Programming Language :: Python :: 3.15"` and
  `cp315-*` to `[tool.cibuildwheel] build`.
- `.github/workflows/ci.yml`: add `"3.15"` to the `test` matrix and
  `allow-prereleases: true` to its `setup-python` step (drop the flag once 3.15
  is final).

Deferred during the fleet-wide Python 3.15 pass (2026-08-18), where pylu, pydgq,
chandra and arxiv-api-search all went green.
