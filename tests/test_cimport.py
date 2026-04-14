"""Verify that wlsqm's Cython declarations (.pxd files) are installed.

Downstream Cython packages rely on `cimport wlsqm.fitter.defs`,
`cimport wlsqm.utils.lapackdrivers`, etc. — which only works if the
.pxd files are shipped alongside the compiled extensions. Meson installs
them via explicit `py.install_sources(...)` calls in each subpackage's
meson.build; this test guards against those entries being lost.
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


FITTER_PXDS = [
    "defs.pxd",
    "infra.pxd",
    "impl.pxd",
    "polyeval.pxd",
    "interp.pxd",
    "simple.pxd",
]

UTILS_PXDS = [
    "lapackdrivers.pxd",
    "ptrwrap.pxd",
]


def test_fitter_pxds_installed():
    import wlsqm.fitter
    pkg_dir = Path(wlsqm.fitter.__file__).parent
    missing = [p for p in FITTER_PXDS if not (pkg_dir / p).exists()]
    assert not missing, f"Missing in {pkg_dir}: {missing}"


def test_utils_pxds_installed():
    import wlsqm.utils
    pkg_dir = Path(wlsqm.utils.__file__).parent
    missing = [p for p in UTILS_PXDS if not (pkg_dir / p).exists()]
    assert not missing, f"Missing in {pkg_dir}: {missing}"


def test_version_file_installed():
    import wlsqm
    version_file = Path(wlsqm.__file__).parent / "VERSION"
    assert version_file.exists()
    assert version_file.read_text().strip() == wlsqm.__version__


@pytest.mark.parametrize(
    "cimport_line",
    [
        "cimport wlsqm.fitter.defs as defs",
        "cimport wlsqm.fitter.infra as infra",
        "cimport wlsqm.fitter.polyeval as polyeval",
        "cimport wlsqm.fitter.interp as interp",
        "cimport wlsqm.fitter.impl as impl",
        "cimport wlsqm.fitter.simple as simple",
        "cimport wlsqm.utils.lapackdrivers as drivers",
        "cimport wlsqm.utils.ptrwrap as ptrwrap",
    ],
)
def test_cimport_compiles(cimport_line):
    """Each .pxd must be discoverable by Cython at compile time — not just
    present as a file. This builds a tiny .pyx containing the single
    cimport and asks cython to compile it; failures point at missing .pxd
    install entries in meson.build or stale install paths."""
    pytest.importorskip("Cython")
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx = Path(tmpdir) / "test_cimport.pyx"
        pyx.write_text(textwrap.dedent(f"""\
            {cimport_line}
            # If this compiles, the .pxd is discoverable.
        """))
        result = subprocess.run(
            [sys.executable, "-m", "cython", "-3", str(pyx)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"{cimport_line!r} failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
