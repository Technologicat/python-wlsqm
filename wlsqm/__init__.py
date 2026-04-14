"""WLSQM (Weighted Least SQuares Meshless): a fast and accurate meshless least-squares interpolator for Python, for scalar-valued data defined as point values on 1D, 2D and 3D point clouds.

WLSQM constructs a piecewise polynomial surrogate model on a local neighborhood of
scattered data points, by a weighted least-squares fit against a monomial basis.
Despite the basis being the same one that appears in a Taylor expansion, the fit
is NOT a Taylor series: the coefficients come from a least-squares solve, and the
averaging effect makes the error behavior much better than Taylor truncation
would predict. For the full theory, see `doc/wlsqm_gen.pdf`.

A general overview is in the README. For the API, see `wlsqm.fitter.simple` and
`wlsqm.fitter.expert`.

When imported, this module re-exports the public names of the following submodules,
so they are available as e.g. `wlsqm.fit_2D(...)`, `wlsqm.ExpertSolver`, ...:

    wlsqm.fitter.defs    # named constants (algorithms, weightings, DOF indices, bitmasks)
    wlsqm.fitter.simple  # simple fit API
    wlsqm.fitter.interp  # interpolation of fitted model
    wlsqm.fitter.expert  # advanced API
"""

from pathlib import Path as _Path
__version__ = (_Path(__file__).parent / "VERSION").read_text().strip()

from .fitter.defs   import *  # noqa: F401, F403 -- re-export submodule public API
from .fitter.simple import *  # noqa: F401, F403
from .fitter.interp import *  # noqa: F401, F403
from .fitter.expert import *  # noqa: F401, F403
