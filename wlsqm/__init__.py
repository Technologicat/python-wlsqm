# -*- coding: utf-8 -*-
#
"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

A general overview can be found in the README.

For the API, refer to  wlsqm.fitter.simple  and  wlsqm.fitter.expert.

When imported, this module imports all symbols from the following modules to the local namespace:

    wlsqm.fitter.defs    # definitions (constants) (common)
    wlsqm.fitter.simple  # simple API
    wlsqm.fitter.interp  # interpolation of fitted model (for simple API)
    wlsqm.fitter.expert  # advanced API

This makes the names available as wlsqm.fit_2D(), wlsqm.ExpertSolver, etc.

JJ 2017-02-22
"""

# absolute_import: https://www.python.org/dev/peps/pep-0328/
from __future__ import division, print_function, absolute_import

__version__ = '0.1.6'

from .fitter.defs   import *  # definitions (constants) (common)
from .fitter.simple import *  # simple API
from .fitter.interp import *  # interpolation of fitted model (for simple API)
from .fitter.expert import *  # advanced API

