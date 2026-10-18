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


def _switch(t: float) -> tuple[float, float]:
    """S(t) and dS/dt of the C^2 quintic switch."""
    return 1 - 10 * t**3 + 15 * t**4 - 6 * t**5, -30 * t**2 * (1 - t) ** 2


def test_ljpot_switch_width_matches_closed_form():
    """LJPot(switch_width=w) is the unshifted 12-6 energy times S(t)."""
    u0, psi, cutoff, width, r = 1.0, 1.0, 2.5, 0.5, 2.25
    positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0

    pot = rgpot.LJPot(u0=u0, cutoff=cutoff, psi=psi, switch_width=width)
    energy, forces, _ = pot(positions, atom_types, box)

    v = 4 * u0 * ((psi / r) ** 12 - (psi / r) ** 6)
    dv = 4 * u0 * (-12 * psi**12 / r**13 + 6 * psi**6 / r**7)
    s, ds_dt = _switch((r - (cutoff - width)) / width)
    assert energy == pytest.approx(v * s, rel=1e-12)
    # Atom 0 sits at x = 0, atom 1 at x = r: F_0 = dE/dr.
    assert forces[0, 0] == pytest.approx(dv * s + v * ds_dt / width, rel=1e-10)
    assert forces[1, 0] == pytest.approx(-forces[0, 0], rel=1e-12)

    shifted, _, _ = rgpot.LJPot(u0=u0, cutoff=cutoff, psi=psi)(
        positions, atom_types, box
    )
    assert shifted != pytest.approx(energy, rel=1e-6)
    with pytest.raises(ValueError):
        rgpot.LJPot(cutoff=2.5, switch_width=3.0)


def test_ljpot_defaults_are_the_cpp_config():
    positions = np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 40.0
    e_default, _, _ = rgpot.LJPot()(positions, atom_types, box)
    e_explicit, _, _ = rgpot.LJPot(u0=1.0, cutoff=15.0, psi=1.0, switch_width=0.0)(
        positions, atom_types, box
    )
    assert e_default == e_explicit


@pytest.mark.skipif(
    getattr(rgpot, "ExprPot", None) is None, reason="built without -Dwith_expr"
)
def test_expr_children_take_pair_config_keys():
    positions = np.array([[0.0, 0.0, 0.0], [2.25, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.zeros(2, dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 30.0
    direct, _, _ = rgpot.LJPot(cutoff=2.5, switch_width=0.5)(
        positions, atom_types, box
    )
    expr = rgpot.ExprPot(
        "lj", {"lj": {"kind": "lj", "cutoff": 2.5, "switch_width": 0.5}}
    )
    via_expr, _, _ = expr(positions, atom_types, box)
    assert via_expr == pytest.approx(direct, rel=1e-12)

    r, De, a, re, cutoff, width = 5.0, 0.7102, 1.6047, 2.8970, 6.0, 2.0
    positions[1, 0] = r
    morse = rgpot.ExprPot(
        "m", {"m": {"kind": "morse", "cutoff": cutoff, "switch_width": width}}
    )
    energy, _, _ = morse(positions, atom_types, box)
    d = 1 - math.exp(-a * (r - re))
    s, _ = _switch((r - (cutoff - width)) / width)
    assert energy == pytest.approx((De * d * d - De) * s, rel=1e-12)


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
