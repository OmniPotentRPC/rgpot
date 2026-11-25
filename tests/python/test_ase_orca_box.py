# MIT License
# Copyright 2023--present rgpot developers
"""The box RgpotCalculator hands a cluster passes ORCAPot's box rule.

ORCAPot has no periodic boundary conditions and takes a box only as padding
around a cluster. It accepts a box when, along each cell vector, the atoms
span at most half the cell width, so no atom would wrap away from the rest
of the cluster (``checkClusterBox`` in CppCore/rgpot/ORCAPot/ORCAPot.cc).
ORCAPot has no Python binding, so a stand-in potential that is not periodic,
like ORCAPot, applies the same rule to every box the calculator gives it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ase import Atoms

from rgpot.ase import RgpotCalculator

WATER = [
    [0.0, 0.0, 0.1173],
    [0.0, 0.7572, -0.4692],
    [0.0, -0.7572, -0.4692],
]


def orca_box_accepts(positions: np.ndarray, box: np.ndarray) -> bool:
    """ORCAPot's rule: all zero, or a span of at most half of every width."""
    cell = np.asarray(box, dtype=np.float64).reshape(3, 3)
    if not cell.any():
        return True
    volume = abs(float(np.linalg.det(cell)))
    for k in range(3):
        normal = np.cross(cell[(k + 1) % 3], cell[(k + 2) % 3])
        length = float(np.linalg.norm(normal))
        width = volume / length
        extent = float(np.ptp(positions @ normal / length))
        if 2.0 * extent > width * (1.0 + 1e-12):
            return False
    return True


class ClusterOnly:
    """Not periodic, no stress, no cutoff; records what it is handed."""

    cutoff = 0.0

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def caps(self) -> dict[str, Any]:
        return {"periodic": False, "stress": False}

    def __call__(
        self, positions: Any, numbers: Any, box: Any
    ) -> tuple[float, np.ndarray, float]:
        pos = np.array(positions, dtype=np.float64).reshape(-1, 3)
        cell = np.array(box, dtype=np.float64).reshape(3, 3)
        if not orca_box_accepts(pos, cell):
            raise ValueError("the atoms would wrap in this box")
        self.calls.append((pos, cell))
        return 0.0, np.zeros_like(pos), 0.0


def _evaluate(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    pot = ClusterOnly()
    atoms.calc = RgpotCalculator(pot)
    atoms.get_potential_energy()
    assert len(pot.calls) == 1
    return pot.calls[0]


@pytest.mark.parametrize(
    "shift",
    [(0.0, 0.0, 0.0), (-140.0, 37.5, 0.0), (0.0, 0.0, -49.9)],
    ids=["origin", "far", "below"],
)
def test_water_cluster_box_passes(shift: tuple[float, float, float]) -> None:
    positions = np.array(WATER) + np.array(shift)
    atoms = Atoms("OH2", positions=positions, pbc=False)
    pos, box = _evaluate(atoms)
    # The atoms are passed on where they are, not wrapped into the box.
    np.testing.assert_array_equal(pos, positions)
    np.testing.assert_array_equal(box, np.eye(3) * 100.0)
    assert orca_box_accepts(pos, box)


def test_caller_cell_is_replaced_by_the_padded_box() -> None:
    atoms = Atoms("OH2", positions=WATER, cell=[2.0, 2.0, 2.0], pbc=False)
    # The caller's cell alone would wrap the molecule.
    assert not orca_box_accepts(np.array(WATER), np.eye(3) * 2.0)
    _, box = _evaluate(atoms)
    np.testing.assert_array_equal(box, np.eye(3) * 100.0)


def test_wide_cluster_box_is_twice_its_extent() -> None:
    atoms = Atoms("He2", positions=[[-30.0, 0.0, 0.0], [30.0, 0.0, 0.0]], pbc=False)
    pos, box = _evaluate(atoms)
    np.testing.assert_array_equal(box, np.eye(3) * 120.0)
    assert orca_box_accepts(pos, box)


def test_split_cluster_is_refused() -> None:
    split = np.array([[1.0, 50.0, 50.0], [99.0, 50.0, 50.0]])
    assert not orca_box_accepts(split, np.eye(3) * 100.0)
    assert orca_box_accepts(split, np.eye(3) * 196.0)


def test_periodic_atoms_are_refused_before_the_kernel() -> None:
    pot = ClusterOnly()
    atoms = Atoms("OH2", positions=WATER, cell=[10.0, 10.0, 10.0], pbc=True)
    atoms.calc = RgpotCalculator(pot)
    with pytest.raises(ValueError, match="set pbc=False"):
        atoms.get_potential_energy()
    assert pot.calls == []
