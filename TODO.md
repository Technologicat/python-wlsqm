TODO
====

Ambient project TODOs — things that would be nice to do but are not blocking
any particular release. For items I noticed *during* the v1.0 modernization
pass and decided to defer, see `TODO_DEFERRED.md`.

General
=======

- figure out a way to automatically add function signatures to docstrings
  for functions defined in Cython modules
  - in the current docstrings, the "def" line has been simply manually
    copy-pasted into the docstring, causing duplication and potential
    drift between the signature and its documentation.

- move `examples/sudoku_lhs.py` into a separate proper library.


Documentation (theory PDFs in `doc/`)
=====================================

Update the theory PDFs under `doc/` so they match the current code and
current understanding:

1. Emphasize surrogate models / response surface modeling. Drop the
   Taylor-series framing — it is misleading, since it severely
   overestimates the error. The coefficients come from a weighted
   least-squares fit, not from analytic differentiation at the origin,
   and the averaging effect gives much better error behavior than Taylor
   truncation would predict. See the comment block at the top of
   `wlsqm/fitter/polyeval.pyx` for the current framing.

2. Introduce the weighting factors `w[k]`. Change the definition of the
   total squared error `G` to be the weighted total squared error, where
   each neighbor point `x[k]` has its own weight `w[k]`. The end result
   is to weight, in each sum over k, each term by `w[k]`. See
   `wlsqm/fitter/impl.pyx` for the current implementation.

3. Add a section on matrix scaling, which drastically improves the
   condition number of the problem matrix. Include a short explanation
   of how the row and column scaling arrays are used in the solve step.
   Cite the algorithm papers (see `wlsqm/utils/lapackdrivers.pyx`, the
   `rescale_ruiz2001_c` comment has the full reference: Daniel Ruiz,
   *A Scaling Algorithm to Equilibrate Both Rows and Columns Norms in
   Matrices*, Report RAL-TR-2001-034, 2001).

4. Add a section on iterative refinement for roundoff mitigation (this
   technique is standard in least-squares fitting). The use of FMA in
   `wlsqm/fitter/polyeval.pyx`, used internally to compute the residual
   `error = data − model`, further reduces roundoff because it rounds
   only the end result of `op1*op2 + op3`.

5. Combine the three pieces (`wlsqm.pdf`, `wlsqm_gen.pdf`, `eulerflow.pdf`)
   into a single document. The current fragmentation dates back to the
   organic growth of the original FREYA-era writeup.

The documentation pass is a potential arXiv tutorial target.


wlsqm.utils.lapackdrivers
=========================

- Add an option to return the orthogonal matrices `U` and `V` from
  `svd_c()`, which is currently only useful for computing the 2-norm
  condition number. Exposing the full decomposition would let downstream
  code use the driver for more general least-squares problems.


wlsqm.fitter — API professionalism
==================================

- Make `wlsqm.fitter.expert.ExpertSolver` instances copyable.
  - Needs a `copy()` method that deep-copies the C-level allocations
    (re-running the memory-allocation dance).
  - `wlsqm.fitter.infra` would need a `Case_copy()` that copies the
    struct and all the pointers it holds into fresh buffers.

- Make `wlsqm.fitter.expert.ExpertSolver` instances picklable. Same C-
  level story as above.

- Introduce `DTYPE` / `DTYPE_t` aliases instead of using `double` /
  `np.float64` directly, to allow compiling a version with complex-
  number support (Cython's fused types could serve here).


wlsqm.fitter — testing and polish
=================================

- More 3D coverage. Extend the test suite with randomly generated SymPy
  polynomials (seed = 42), differentiated symbolically, fitted at orders
  0–4, with all ~34 derivatives compared against the exact result.
  Target: worst-case error within ~100·machine epsilon for the function
  value, tightening with lower derivative order.

- Profile performance.
  [http://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line](http://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line)

- Various small TODOs and FIXMEs in the code (low priority; grep for
  `TODO` / `FIXME` in the sources).


wlsqm.fitter — API ergonomics
=============================

- `ExpertSolver`: allow interpolating the model to a single point without
  a memoryview slice. Currently the input must be a memoryview because
  the general case is non-contiguous; single-point usage should not
  require the caller to build one. Profile first to check whether the
  current path is a real bottleneck.

- Reduce duplication between driver mode and expert mode: split
  `generic_fit_basic_many()` and friends into prepare and solve stages,
  and have `ExpertSolver` call the same stages. The driver mode would
  then be a thin wrapper. Care needed: the driver mode's behavior (a
  one-shot fit) must stay identical.
