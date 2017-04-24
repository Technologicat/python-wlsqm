# -*- coding: utf-8 -*-
#
"""Setuptools-based setup script for WLSQM."""

from __future__ import division, print_function, absolute_import

#########################################################
# Config
#########################################################

# choose build type here
#
build_type="optimized"
#build_type="debug"


#########################################################
# Init
#########################################################

# check for Python 2.7 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (2,7):
    sys.exit('Sorry, Python < 2.7 is not supported')

import os

from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules for WLSQM.")


#########################################################
# Definitions
#########################################################

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
    print( "build configuration selected: optimized" )
else: # build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args            = extra_link_args_debug
    debug = True
    print( "build configuration selected: debug" )


#########################################################
# Long description
#########################################################

DESC="""WLSQM (Weighted Least SQuares Meshless) is a fast and accurate meshless least-squares interpolator for Python, implemented in Cython.

Given scalar data values on a set of points in 1D, 2D or 3D, WLSQM constructs a piecewise polynomial global surrogate model (a.k.a. response surface), using up to 4th order polynomials.

Use cases include response surface modeling, and computing space derivatives of data known only as values at discrete points in space. No grid or mesh is needed.

Any derivative of the model function (e.g. d2f/dxdy) can be easily evaluated, up to the order of the polynomial.

Sensitivity data of solution DOFs (on the data values at points other than the reference in the local neighborhood) can be optionally computed.

Performance-critical parts are implemented in Cython. LAPACK is used via SciPy's Cython-level bindings. OpenMP is used for parallelization over the independent local problems (also in the linear solver step).

This implementation is targeted for high performance in a single-node environment, such as a laptop. The main target is x86_64.
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

# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datadirs  = ("examples",)
dataexts  = (".py", ".pyx", ".pxd", ".c", ".sh", ".lyx", ".pdf")
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
    datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                       for root, dirs, files in os.walk(datadir)] )

datafiles.append( ('.', ["README.md", "LICENSE.md", "TODO.md", "CHANGELOG.md"]) )


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

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
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
        print( "WARNING: Version information not found, using placeholder '%s'" % (version) )


setup(
    name = "wlsqm",
    version = version,
    author = "Juha Jeronen",
    author_email = "juha.jeronen@jyu.fi",
    url = "https://github.com/Technologicat/python-wlsqm",

    description = "Weighted least squares meshless interpolator",
    long_description = DESC,

    license = "BSD",
    platforms = ["Linux"],  # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does

    classifiers = [ "Development Status :: 4 - Beta",
                    "Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: BSD License",
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
    setup_requires = ["cython", "scipy (>=0.16)"],
    install_requires = ["numpy", "scipy (>=0.16)"],
    provides = ["wlsqm"],

    # same keywords as used as topics on GitHub
    keywords = ["numerical interpolation differentiation curve-fitting least-squares meshless numpy cython"],

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
    # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
    package_data={'wlsqm.utils': ['*.pxd', '*.pyx'],  # note: paths relative to each package
                  'wlsqm.fitter': ['*.pxd', '*.pyx']},

    # Disable zip_safe, because:
    #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
    #   - dynamic loader may need to have the library unzipped to a temporary folder anyway (at import time)
    zip_safe = False,

    # Usage examples; not in a package
    data_files = datafiles
)
