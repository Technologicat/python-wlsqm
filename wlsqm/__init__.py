# -*- coding: utf-8 -*-
#
"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

A general overview can be found in the README.

For the API, see  fitter  and  wlsqm2_expert.

JJ 2017-02-22
"""

from __future__ import absolute_import  # https://www.python.org/dev/peps/pep-0328/

__version__ = '0.1.0'

from .wlsqm2.wlsqm2_defs   import *  # definitions (constants) (common)
from .wlsqm2.fitter        import *  # simple API
from .wlsqm2.wlsqm2_eval   import *  # interpolation of fitted model (for simple API)
from .wlsqm2.wlsqm2_expert import *  # advanced API

