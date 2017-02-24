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

# extract __version__ from the package __init__.py (it's not a good idea to actually run it during the build process)
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


# TODO: add long_description, based on README
# TODO: add url, download_url (this project has no real homepage...)
setup(
    name = "wlsqm",
    version = version,
    author = "Juha Jeronen",
    author_email = "juha.jeronen@jyu.fi",

    description = "WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.",

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

    packages = ["wlsqm", "wlsqm.utils", "wlsqm.fitter"],  # packages must be declared so that  python -m setup build  will copy .py files
    package_data={'wlsqm.utils': ['*.pxd'],  # note: paths relative to each package
                  'wlsqm.fitter': ['*.pxd']},
)

