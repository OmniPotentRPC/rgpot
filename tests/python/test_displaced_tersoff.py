"""Displaced Tersoff geometries used by the asv rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "openmp_gather", _ROOT / "benchmarks" / "openmp_gather.py"
)
_GATHER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_GATHER)


def test_displaced_silicon_is_deterministic_and_off_the_lattice() -> None:
    lattice, _, box = _GATHER._fcc(64)
    pos, types, shifted = _GATHER.displaced_fcc(64, 14)
    again, _, _ = _GATHER.displaced_fcc(64, 14)
    assert np.array_equal(pos, again)
    assert np.array_equal(box, shifted)
    assert types.shape == (64,)
    assert set(types.tolist()) == {14}
    delta = pos - lattice
    rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    assert abs(rms - 0.05) < 0.01
    assert not np.array_equal(pos, lattice)


def test_displaced_silicon_carbide_alternates_species() -> None:
    silicon, _, _ = _GATHER.displaced_fcc(256, 14)
    mixed, types, _ = _GATHER.displaced_fcc(256, _GATHER.silicon_carbon(256))
    assert np.array_equal(silicon, mixed)
    assert int(np.sum(types == 14)) == 128
    assert int(np.sum(types == 6)) == 128
    assert list(types[:4]) == [14, 6, 14, 6]


def test_asv_rows_cover_the_displaced_sizes() -> None:
    assert _GATHER.TersoffDisplacedSi.params[0] == [64, 256, 1024, 4096]
    assert _GATHER.TersoffDisplacedSiC.params[0] == [64, 256, 1024, 4096]
    assert _GATHER.TersoffDisplacedSi.param_names == ["atoms", "threads"]
    si = _GATHER.TersoffDisplacedSi()
    mix = _GATHER.TersoffDisplacedSiC()
    pos, types, _box = si._geometry(64)
    _, mixed, _ = mix._geometry(64)
    assert types.tolist() == [14] * 64
    assert mixed.tolist()[1::2] == [6] * 32
    assert pos.shape == (64, 3)
