High priority
=============

- create unit tests

-------------------------------------------------------------------------------

General
=======

- figure out a way to automatically add function signatures to docstrings for functions defined in Cython modules
  - in the current docstrings, the "def" has been simply manually copy'n'pasted into the docstring,
    causing unnecessary duplication and introducing a potential source of errors in documentation.

- move examples/sudoku_lhs into a separate proper library.

Documentation
=============

- Update the documentation:
  1. Emphasize surrogate models / response surface modeling (the Taylor series based intuition is misleading, as it severely overestimates the error).
  2. Introduce the weighting factors `w[k]`. Change the definition of the total squared error G to be the weighted total squared error, where each neighbor point x[k] has its own weight w[k]. The end result is basically just to weight, in each sum over k, each term by `w[k]`. See `wlsqm/fitter/impl.pyx`.
  3. Add a comment about matrix scaling, which drastically improves the condition number. Include a short comment on how to use the row and column scaling arrays. Cite the algorithm papers (see `wlsqm/utils/lapackdrivers.pyx`).
  4. Add a comment about iterative refinement to reduce effects of roundoff (this technique is rather standard in least-squares fitting). The use of FMA in `wlsqm/fitter/polyeval.pyx`, used internally to compute `error = (data - model)`, may also mitigate roundoff, since it computes `op1*op2 + op3`, rounding only the end result.
  5. Combine the pieces into a single document (see README.md for a listing of the pieces).

utils.lapackdrivers
===================

 - add option to return also the orthogonal matrices U and V in `svd()` (currently this routine is only useful to compute the 2-norm condition number)

fitter
======

 - fix TODOs in `setup.py`

 - API professionalism:
   - make `wlsqm.fitter.expert.ExpertSolver` instances copyable
     - needs a copy() method that deep-copies also the C-level stuff (re-running the memory allocation fandango)
     - `wlsqm.fitter.infra` needs a `Case_copy()` method, because `wlsqm.fitter.infra.Case` contains pointers
   - make `wlsqm.fitter.expert.ExpertSolver` instances pickleable (need to save/load the C-level stuff)
   - use `DTYPE` and `DTYPE_t` aliases instead of `double`/`np.float64` directly, to allow compiling a version with complex number support

 - test the 3D support more thoroughly
   - `wlsqm/fitter/polyeval.pyx`: make really, really sure `taylor_3D()`, `general_3D()` are bug-free
   - `wlsqm/fitter/interp.pyx`: make really, really sure `interpolate_3D()` is bug-free
   - write a unit test: generate random `sympy` functions (from a preset seed to make the test repeatable), differentiate them symbolically, fit models of orders 0, 1, 2, 3, 4 and compare all up to 34 derivatives with the exact result (the worst case should be within approx. `100*machine_epsilon` at least for the function value itself).

 - profile performance, see [http://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line](http://stackoverflow.com/questions/28301931/how-to-profile-cython-functions-line-by-line)

 - fix various small TODOs and FIXMEs in the code (low priority)

 - maybe: ExpertSolver: fix the silly slicing requirement in model interpolation: make it possible to interpolate the model to a single point without a memoryview
   - but profile the performance first to check whether this actually causes a problem
   - multiple points require the memoryview, because in the general case the input is non-contiguous (a sliced array)

 - maybe: reduce code duplication between driver and expert mode
   - split `generic_fit_basic_many()` (and its friends) into prepare and solve stages, implement the driver in terms of calling these stages
   - re-use the same stages in ExpertSolver

