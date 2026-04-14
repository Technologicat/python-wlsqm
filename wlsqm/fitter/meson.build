# include_directories('.') so infra.pyx can #include "popcount.h"
# (a small MSVC-compat shim sitting alongside the source).
fitter_inc = include_directories('.')

# defs — constants (Python-accessible + C-level); leaf module.
py.extension_module(
    'defs',
    'defs.pyx',
    dependencies: [numpy_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

# infra — Allocator / CaseManager / Case structs + memory management.
# Needs popcount.h via include_directories.
py.extension_module(
    'infra',
    'infra.pyx',
    dependencies: [numpy_dep, m_dep],
    include_directories: fitter_inc,
    install: true,
    subdir: 'wlsqm/fitter',
)

# polyeval — Horner-form polynomial evaluation with fused multiply-add.
py.extension_module(
    'polyeval',
    'polyeval.pyx',
    dependencies: [numpy_dep, m_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

# interp — interpolation of a fitted local polynomial model.
py.extension_module(
    'interp',
    'interp.pyx',
    dependencies: [numpy_dep, m_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

# impl — low-level routines (distance matrix, A assembly, solve). Uses OpenMP.
py.extension_module(
    'impl',
    'impl.pyx',
    dependencies: [numpy_dep, m_dep, omp_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

# simple — simple fit API (single case + many-cases serial/parallel). Uses OpenMP.
py.extension_module(
    'simple',
    'simple.pyx',
    dependencies: [numpy_dep, m_dep, omp_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

# expert — ExpertSolver (prepare/solve stages, advanced API). Uses OpenMP.
py.extension_module(
    'expert',
    'expert.pyx',
    dependencies: [numpy_dep, m_dep, omp_dep],
    install: true,
    subdir: 'wlsqm/fitter',
)

py.install_sources(
    '__init__.py',
    subdir: 'wlsqm/fitter',
)

# Install .pxd files for downstream cimport.
py.install_sources(
    'defs.pxd',
    'infra.pxd',
    'impl.pxd',
    'polyeval.pxd',
    'interp.pxd',
    'simple.pxd',
    subdir: 'wlsqm/fitter',
)
