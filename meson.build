project(
    'wlsqm',
    'cython', 'c',
    version: files('wlsqm/VERSION'),
    license: 'BSD-2-Clause',
    meson_version: '>=1.3.0',
    default_options: [
        'buildtype=release',
        'c_std=c11',
        'cython_language=c',
    ],
)

# Expose M_PI etc. on MSVC. No-op on Linux/macOS.
add_project_arguments('-D_USE_MATH_DEFINES', language: 'c')

# Global Cython language level. Replaces per-file `# cython: language_level = 3`
# directives (which wlsqm doesn't currently use at all).
add_project_arguments('-X', 'language_level=3', language: 'cython')

py = import('python').find_installation(pure: false)

# NumPy include directory for Cython compilation against the NumPy C API.
#
# NumPy 2.0 ships `numpy-config` and meson's `dependency('numpy')` resolves
# it. NumPy 1.x does not. Fall back to the Python introspection approach so
# that users whose stacks are still pinned to NumPy 1.x (e.g. dependencies
# that have not migrated to 2.x yet) can still build wlsqm from source.
numpy_dep = dependency('numpy', required: false)
if not numpy_dep.found()
    numpy_include = run_command(
        py,
        ['-c', 'import numpy; print(numpy.get_include())'],
        check: true,
    ).stdout().strip()
    message('Using NumPy < 2.0 headers from: ' + numpy_include)
    numpy_dep = declare_dependency(include_directories: include_directories(numpy_include))
endif

# SciPy ships `scipy/linalg/cython_lapack.pxd`, which `lapackdrivers.pyx`
# cimports at compile time. SciPy does NOT publish a pkg-config/cmake module,
# so `dependency('scipy')` fails. Instead we resolve the parent of the scipy
# package (i.e. the site-packages dir that contains `scipy/`) at configure
# time and hand it to Cython via a `-I` include path. This is the same trick
# SciPy itself uses internally.
scipy_parent = run_command(
    py,
    ['-c', 'import os, scipy; print(os.path.dirname(scipy.__path__[0]))'],
    check: true,
).stdout().strip()
message('Using SciPy Cython headers from: ' + scipy_parent)
scipy_cython_args = ['-I', scipy_parent]

# libm: linked on Linux/macOS (-lm), no-op on Windows (math is in MSVCRT).
cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)

# OpenMP — required:false so wlsqm still builds serial on platforms without it
# (notably Apple Clang with no libomp installed). When absent, Cython's `prange`
# silently degrades to `range`, and we also define WLSQM_NO_OPENMP so the small
# number of `omp_get_thread_num()` call sites can be guarded if needed.
omp_dep = dependency('openmp', required: false)
if omp_dep.found()
    message('OpenMP found — building with parallel support.')
else
    message('OpenMP NOT found — building serial-only. Parallel fit routines will still run but on a single thread.')
    add_project_arguments('-DWLSQM_NO_OPENMP', language: 'c')
endif

subdir('wlsqm')
