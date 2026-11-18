"""EAMTablePot reads a DYNAMO table through table_path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rgpot

DATA = Path(__file__).resolve().parents[2] / "CppCore" / "tests" / "data" / "eam_table"


def _sheared():
    """The LAMMPS eam/alloy reference shipped beside the table."""
    box = None
    energy = None
    rows = []
    kinds = []
    for line in (DATA / "cuni_sheared.ref").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if parts[0] == "box":
            box = np.array([float(x) for x in parts[1:]], dtype=float).reshape(3, 3)
        elif parts[0] == "energy":
            energy = float(parts[1])
        elif parts[0] == "atoms":
            continue
        elif parts[0] in {"stress"}:
            continue
        else:
            kinds.append(int(parts[0]))
            rows.append([float(x) for x in parts[1:4]])
    pos = np.array(rows, dtype=float)
    cell = box
    # Same minimum-image fold as EAMTablePotTest.
    for i in range(len(pos)):
        r = pos[i]
        s2 = r[2] / cell[2, 2]
        s1 = (r[1] - s2 * cell[2, 1]) / cell[1, 1]
        s0 = (r[0] - s1 * cell[1, 0] - s2 * cell[2, 0]) / cell[0, 0]
        s = np.array([s0, s1, s2])
        s -= np.floor(s)
        pos[i] = s[0] * cell[0] + s[1] * cell[1] + s[2] * cell[2]
    return pos, kinds, cell, energy


def test_setfl_matches_the_lammps_reference():
    if rgpot.EAMTablePot is None:
        pytest.skip("this build has no EAM table backend")
    pos, kinds, box, energy = _sheared()
    pot = rgpot.EAMTablePot(table_path=str(DATA / "CuNi.eam.alloy"), format="setfl")
    got, forces, _variance = pot(pos, kinds, box)
    assert got == pytest.approx(energy, abs=1e-8)
    assert forces.shape == pos.shape
    assert np.isfinite(forces).all()


def test_unknown_format_is_rejected():
    if rgpot.EAMTablePot is None:
        pytest.skip("this build has no EAM table backend")
    with pytest.raises(ValueError):
        rgpot.EAMTablePot(table_path=str(DATA / "Cu_u3.eam"), format="not-a-format")
