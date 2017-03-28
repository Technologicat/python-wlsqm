# python-wlsqm

WLSQM (Weighted Least SQuares Meshless) is a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

Use cases include response surface modeling, and computing space derivatives of data known only as values at discrete points in space (this has applications in explicit algorithms for solving IBVPs). No grid or mesh is needed. No restriction is imposed on geometry other than "not degenerate", e.g. points in 2D should not all fall onto the same 1D line.

This is an independent implementation of the weighted least squares meshless algorithm described (in the 2nd order 2D case) in section 2.2.1 of Hong Wang (2012), Evolutionary Design Optimization with Nash Games and Hybridized Mesh/Meshless Methods in Computational Fluid Dynamics, Jyväskylä Studies in Computing 162, University of Jyväskylä. [ISBN 978-951-39-5007-1 (PDF)](http://urn.fi/URN:ISBN:978-951-39-5007-1)

This implementation is targeted for high performance in a single-node environment, such as a laptop. Cython is used to accelerate the low-level routines. The main target is the `x86_64` architecture, but any 64-bit architecture should be fine with the appropriate compiler option changes to `setup.py`.

Currently only Python 2.7 is supported, but this may change in the future. Automated unit tests are missing; this is another area that is likely to be improved. Otherwise the code is already rather stable; any major new features are unlikely to be added, and the API is considered stable.


## Features

- Given scalar data values on a set of points in 1D, 2D or 3D, construct a piecewise polynomial global surrogate model (a.k.a. response surface), using up to 4th order polynomials.

- Sliced arrays are supported for input, both for the geometry (points) and data (function values).

- Obtain any derivative of the model, up to the order of the polynomial. Derivatives at each "local model reference point" xi are directly available as DOFs of the solution. Derivatives at any other point can be automatically interpolated from the model. Differentiation of polynomials has been hardcoded to obtain high performance.

- Knowns. At the model reference point xi, the function value and/or any of the derivatives can be specified as knowns. The knowns are internally automatically eliminated (making the equation system smaller) and only the unknowns are fitted. The function value itself may also be unknown, which is useful for implementing Neumann BCs in a PDE (IBVP) solving context.

- Selectable weighting method for the fitting error, to support different use cases:
  - uniform (`wlsqm.fitter.defs.WEIGHT_UNIFORM`), for best overall fit for function values
  - emphasize points closer to xi (`wlsqm.fitter.defs.WEIGHT_CENTER`), to improve derivatives at the reference point xi by reducing the influence of points far away from the reference point.

- Sensitivity data of solution DOFs (on the data values at points other than the reference in the local neighborhood) can be optionally computed.

- Expert mode with separate prepare and solve stages, for faster fitting of many data sets using the same geometry. Also performs global model patching, using the set of local models fitted.

  **CAVEAT**: `wlsqm.fitter.expert.ExpertSolver` instances are not currently pickleable or copyable. This is a known limitation that may (or may not) change in the future.

  It is nevertheless recommended to use ExpertSolver, since this allows for easy simultaneous solving of many local models (in parallel), automatic global model patching, and reuse of problem matrices when the geometry of the point cloud does not change.

- Speed:
  - Performance-critical parts are implemented in Cython, and the GIL is released during computation.
  - LAPACK is used directly via [SciPy's Cython-level bindings](https://docs.scipy.org/doc/scipy/reference/linalg.cython_lapack.html) (see the `ntasks` parameter in various API functions in `wlsqm`). This is especially useful when many (1e4 or more) local models are being fitted, as the solver loop does not require holding the GIL.
  - OpenMP is used for parallelization over the independent local problems (also in the linear solver step).
  - The polynomial evaluation code has been manually optimized to reduce the number of FLOPS required.

    In 1D, the Horner form is used. The 2D and 3D cases use a symmetric form that extends the 1D Horner form into multiple dimensions (see `wlsqm/fitter/polyeval.pyx` for details). The native FMA (fused multiply-add) instruction of the CPU is used in the evaluation to further reduce FLOPS required, and to improve accuracy (utilizing the fact it rounds only once).

- Accuracy:
  - Problem matrices are preconditioned by a symmetry-preserving scaling algorithm (D. Ruiz 2001; exact reference given in `wlsqm/utils/lapackdrivers.pyx`) to obtain best possible accuracy from the direct linear solver. This is critical especially for high-order fits.
  - The fitting procedure optionally accomodates an internal iterative refinement loop to mitigate the effect of roundoff.
  - FMA, as mentioned above.


## Documentation

For usage examples, see `examples/wlsqm_example.py`.

For the technical details, see the docstrings and comments in the code itself.

Mathematics documented at:

  [https://yousource.it.jyu.fi/jjrandom2/freya/trees/master/docs](https://yousource.it.jyu.fi/jjrandom2/freya/trees/master/docs)

where the relevant files are:

  - wlsqm.pdf (old documentation for the old pure-Python version of WLSQM included in FREYA, plus the sensitivity calculation)
  - eulerflow.pdf (clearer presentation of the original version, but without the sensitivity calculation)
  - wlsqm_gen.pdf (theory diff on how to make a version that handles also missing function values; also why WLSQM works and some analysis of its accuracy)

The documentation is slightly out of date; see TODO.md for details on what needs updating and how.


## Experiencing crashes?

Check that you are loading the same BLAS your LAPACK and SciPy link against::
    shopt -s globstar
    ldd /usr/lib/**/*lapack*.so | grep blas
    ldd $(dirname $(python -c "import scipy; print(scipy.__file__)"))/linalg/cython_lapack.so | grep blas

In Debian-based Linux, you can change the active BLAS implementation by::
    sudo update-alternatives --config libblas.so
    sudo update-alternatives --config libblas.so.3

This may (or may not) be different from what NumPy links against::
    ldd $(dirname $(python -c "import numpy; print(numpy.__file__)"))/core/multiarray.so | grep blas

WLSQM itself does not link against LAPACK or BLAS; it utilizes the `cython_lapack` module of SciPy.


## License

BSD (see LICENSE.md). Copyright 2016-2017 Juha Jeronen and University of Jyväskylä.


#### Acknowledgement

This work was financially supported by the Jenny and Antti Wihuri Foundation.

