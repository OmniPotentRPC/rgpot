"""Tests for the pip-installable rgpot core (real package, real LJ force)."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the real package under test — not a reimplementation.
rgpot = pytest.importorskip("rgpot")


def test_version_and_exports():
    assert hasattr(rgpot, "__version__")
    assert rgpot.__version__
    assert callable(rgpot.evaluate_lj)
    assert rgpot.LJPot is not None


def test_evaluate_lj_finite_forces():
    """Two-atom LJ: finite energy and forces with shape (2, 3)."""
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([0, 0], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 20.0

    energy, forces, variance = rgpot.evaluate_lj(positions, atom_types, box)

    assert math.isfinite(float(energy)), f"energy not finite: {energy}"
    assert forces.shape == (2, 3), forces.shape
    assert np.all(np.isfinite(forces)), forces
    assert math.isfinite(float(variance))
    # Opposite forces on a pair (action-reaction) within numerical noise
    assert np.allclose(forces[0] + forces[1], 0.0, atol=1e-9), forces


def test_ljpot_class_matches_evaluate_lj():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float64
    )
    atom_types = np.zeros(3, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0

    e1, f1, v1 = rgpot.evaluate_lj(positions, atom_types, box)
    pot = rgpot.LJPot()
    e2, f2, v2 = pot(positions, atom_types, box)

    assert abs(e1 - e2) < 1e-12
    assert np.allclose(f1, f2)
    assert abs(v1 - v2) < 1e-12


def test_extension_rpath_has_no_host_build_tree():
    """Installed _core must not embed absolute build-host RUNPATH entries."""
    parent = Path(rgpot.__file__).resolve().parent
    cores = list(parent.glob("_core*.so")) + list(parent.glob("_core*.pyd"))
    assert cores, f"missing _core extension under {parent}"
    so = cores[0]
    if not shutil.which("readelf"):
        pytest.skip("readelf not available")
    out = subprocess.check_output(["readelf", "-d", str(so)], text=True)
    # Stricter: no build-tree / home build markers in RPATH/RUNPATH lines
    for line in out.splitlines():
        low = line.lower()
        if "runpath" not in low and "rpath" not in low:
            continue
        for bad in ("/home/", "/Users/", "/bbdir", "/build-pyeon", "/.mesonpy-", "/scratch/tmp"):
            assert bad not in line, f"host path marker {bad!r} in:\n{line}\nfull:\n{out}"


@pytest.mark.skipif(
    os.environ.get("RGPOT_SKIP_WHEEL_TEST") == "1",
    reason="RGPOT_SKIP_WHEEL_TEST=1",
)
def test_import_with_empty_ld_library_path():
    code = (
        "import os; os.environ.pop('LD_LIBRARY_PATH', None); "
        "os.environ.pop('LIBRARY_PATH', None); "
        "import rgpot; import numpy as np; "
        "e,f,v=rgpot.evaluate_lj("
        "np.array([[0.,0.,0.],[1.5,0.,0.]]),"
        "np.array([0,0],dtype=np.int32),"
        "np.eye(3)*20); "
        "assert f.shape==(2,3); print('RGPOT_FORCE_OK', float(e))"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_LIBRARY_PATH", "LIBRARY_PATH")
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RGPOT_FORCE_OK" in proc.stdout


def test_core_is_abi3_or_free_threaded():
    """On CPython >= 3.12 GIL builds the extension must be abi3-tagged."""
    import sys
    import sysconfig

    parent = Path(rgpot.__file__).resolve().parent
    cores = list(parent.glob("_core*.so")) + list(parent.glob("_core*.pyd"))
    assert cores, f"missing _core under {parent}"
    name = cores[0].name
    gil_disabled = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    if sys.version_info >= (3, 12) and not gil_disabled:
        assert "abi3" in name, f"expected abi3 extension, got {name}"


@pytest.mark.skipif(not rgpot.has_cache, reason="built without -Dwith_cache")
def test_cache_counts_split_computed_from_served(tmp_path):
    """Cold run computes, warm run on the same store is served."""
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([0, 0], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 20.0
    store = str(tmp_path / "cache")

    cold = rgpot.PotentialCache(store)
    pot = rgpot.LJPot()
    pot.set_cache(cold)
    assert cold.counts() == (0, 0)
    first = pot(positions, atom_types, box)
    pot(positions, atom_types, box)
    assert cold.counts() == (1, 1)
    del pot, cold

    warm = rgpot.PotentialCache(store)
    pot = rgpot.LJPot()
    pot.set_cache(warm)
    again = pot(positions, atom_types, box)
    assert warm.counts() == (0, 1)
    assert float(again[0]) == float(first[0])
    warm.reset_counts()
    assert warm.counts() == (0, 0)
