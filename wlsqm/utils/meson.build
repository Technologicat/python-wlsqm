# ptrwrap — tiny pointer-wrapping cdef class. No math, no OpenMP.
py.extension_module(
    'ptrwrap',
    'ptrwrap.pyx',
    dependencies: [numpy_dep],
    install: true,
    subdir: 'wlsqm/utils',
)

# lapackdrivers — LAPACK wrappers (via scipy.linalg.cython_lapack) + OpenMP
# parallel solvers over batches of independent linear systems.
py.extension_module(
    'lapackdrivers',
    'lapackdrivers.pyx',
    dependencies: [numpy_dep, m_dep, omp_dep],
    cython_args: scipy_cython_args,
    install: true,
    subdir: 'wlsqm/utils',
)

py.install_sources(
    '__init__.py',
    subdir: 'wlsqm/utils',
)

# Install .pxd files for downstream cimport. lapackdrivers.pxd lets external
# Cython code plug straight into the parallel LAPACK batch solvers.
py.install_sources(
    'ptrwrap.pxd',
    'lapackdrivers.pxd',
    subdir: 'wlsqm/utils',
)
