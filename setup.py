# -*- coding: utf-8 -*-
#
"""Distutils-based setup script for WLSQM.

Requires Cython.

JJ 2017-02-24
"""

from __future__ import absolute_import

# check for Python 2.7 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

import os

from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    print "Cython not found. Cython is needed to build the extension modules for WLSQM."
    sys.exit(1)


build_type="optimized"
#build_type="debug"


extra_compile_args_math_optimized    = ['-fopenmp', '-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-fopenmp', '-march=native', '-O0', '-g']

extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']

extra_link_args_optimized    = ['-fopenmp']
extra_link_args_debug        = ['-fopenmp']


if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args            = extra_link_args_optimized
    debug = False
    print "build configuration selected: optimized"
else: # build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args            = extra_link_args_debug
    debug = True
    print "build configuration selected: debug"


#########################################################
# Long description
#########################################################

DESC="""WLSQM (Weighted Least SQuares Meshless) is a fast and accurate meshless least-squares interpolator for Python, implemented in Cython.

Given scalar data values on a set of points in 1D, 2D or 3D, WLSQM constructs a piecewise polynomial global surrogate model (a.k.a. response surface), using up to 4th order polynomials.

This is an independent implementation of the weighted least squares meshless algorithm described (in the 2nd order 2D case) in section 2.2.1 of Hong Wang (2012), Evolutionary Design Optimization with Nash Games and Hybridized Mesh/Meshless Methods in Computational Fluid Dynamics, Jyväskylä Studies in Computing 162, University of Jyväskylä. ISBN 978-951-39-5007-1 (PDF). http://urn.fi/URN:ISBN:978-951-39-5007-1

Use cases include response surface modeling, and computing space derivatives of data known only as values at discrete points in space (this has applications in explicit algorithms for solving initial boundary value problems (IBVPs)). No grid or mesh is needed. No restriction is imposed on geometry other than "not degenerate", e.g. points in 2D should not all fall onto the same 1D line.

Any derivative of the model function (e.g. d2f/dxdy) can be easily evaluated, up to the order of the polynomial. Derivatives at each "local model reference point" xi are directly available as DOFs of the solution. Derivatives at any other point can be automatically interpolated from the model.

Each local model has a reference point, on which the local polynomial will be centered. At the reference point, the function value and/or any of the derivatives can be specified as knowns, in which case they will be automatically eliminated from the equation system. The function value itself may also be unknown (at the reference point only), which is useful for implementing Neumann BCs in a PDE (IBVP) solving context.

Sensitivity data of solution DOFs (on the data values at points other than the reference in the local neighborhood) can be optionally computed.

Sliced arrays are supported for input, both for the geometry (points) and data (function values). Performance-critical parts are implemented in Cython, and the GIL is released during computation. LAPACK is used directly via SciPy's Cython-level bindings. OpenMP is used for parallelization over the independent local problems (also in the linear solver step).

To improve accuracy, problem matrices are preconditioned by a symmetry-preserving scaling algorithm. Fused multiply-add (FMA) is used in polynomial evaluation.

This implementation is targeted for high performance in a single-node environment, such as a laptop. The main target is the x86_64 architecture, but any 64-bit architecture should be fine with the appropriate compiler option changes to setup.py.

Likely future improvements include Python 3 support (currently only Python 2.7 is supported), and automated unit tests. Otherwise the code is already rather stable; any major new features are unlikely to be added, and the public API is considered stable.

Usage examples are provided in the download.
"""

#########################################################
# Helpers
#########################################################

my_include_dirs = ["."]  # IMPORTANT, see https://github.com/cython/cython/wiki/PackageHierarchy

def ext(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_nonmath
                    )
def ext_math(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension( extName,
                      [extPath],
                      extra_compile_args=my_extra_compile_args_math,
                      extra_link_args=my_extra_link_args,
                      libraries=["m"]  # "m" links libm, the math library on unix-likes; see http://docs.cython.org/src/tutorial/external.html
                    )

#########################################################
# Utility modules
#########################################################

ext_module_ptrwrap       = ext(     "wlsqm.utils.ptrwrap")        # Pointer wrapper for Cython/Python integration
ext_module_lapackdrivers = ext_math("wlsqm.utils.lapackdrivers")  # Simple Python interface to LAPACK for solving many independent linear equation systems efficiently in parallel. Built on top of scipy.linalg.cython_lapack.

#########################################################
# WLSQM (Weighted Least SQuares Meshless method)
#########################################################

ext_module_defs     = ext(     "wlsqm.fitter.defs")      # definitions (named constants)
ext_module_infra    = ext(     "wlsqm.fitter.infra")     # memory allocation infrastructure
ext_module_impl     = ext_math("wlsqm.fitter.impl")      # low-level routines (implementation)
ext_module_polyeval = ext_math("wlsqm.fitter.polyeval")  # evaluation of Taylor expansions and general polynomials
ext_module_interp   = ext_math("wlsqm.fitter.interp")    # interpolation of fitted model
ext_module_simple   = ext_math("wlsqm.fitter.simple")    # simple API
ext_module_expert   = ext_math("wlsqm.fitter.expert")    # advanced API

#########################################################

# Extract __version__ from the package __init__.py (it's not a good idea to actually run it during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
#
import ast
with file('wlsqm/__init__.py') as f:
    for line in f:
        if line.startswith('__version__'):
            version = ast.parse(line).body[0].value.s
            break
    else:
        version = '0.0.unknown'
        print "WARNING: Version information not found, using placeholder '%s'" % (version)


# TODO: add download_url
setup(
    name = "wlsqm",
    version = version,
    author = "Juha Jeronen",
    author_email = "juha.jeronen@jyu.fi",
    url = "https://github.com/Technologicat/python-wlsqm",

    description = "WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.",
    long_description = DESC,

    license = "BSD",
    platforms = ["Linux"],  # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does

    classifiers = [ "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: BSD License",
                    "Natural Language :: English",
                    "Operating System :: POSIX :: Linux",
                    "Programming Language :: Cython",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 2",
                    "Programming Language :: Python :: 2.7",
                    "Programming Language :: Python :: 2 :: Only",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Software Development :: Libraries",
                    "Topic :: Software Development :: Libraries :: Python Modules"
                  ],

    # 0.16 seems to be the first SciPy version that has cython_lapack.pxd. ( https://github.com/scipy/scipy/commit/ba438eab99ce8f55220a6ff652500f07dd6a547a )
    requires = ["cython", "numpy", "scipy (>=0.16)"],
    provides = ["wlsqm"],

    cmdclass = {'build_ext': build_ext},

    ext_modules = cythonize( [ ext_module_lapackdrivers,
                               ext_module_ptrwrap,
                               ext_module_defs,
                               ext_module_infra,
                               ext_module_impl,
                               ext_module_polyeval,
                               ext_module_interp,
                               ext_module_simple,
                               ext_module_expert ],

                             include_path = my_include_dirs,

                             gdb_debug = debug ),

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    packages = ["wlsqm", "wlsqm.utils", "wlsqm.fitter"],

    # Install also Cython headers so that other Cython modules can cimport ours
    package_data={'wlsqm.utils': ['*.pxd'],  # note: paths relative to each package
                  'wlsqm.fitter': ['*.pxd']},
)

