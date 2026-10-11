"""Shared geometries for the Python tests."""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms


def lj_dimer(distance: float = 1.5, cell: float = 20.0) -> Atoms:
    """Two atoms along x in a cubic cell."""
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]],
        cell=[cell, cell, cell],
        pbc=False,
    )
    return atoms


def lj_crystal() -> Atoms:
    """Four atoms in a periodic cell, inside the LJ cutoff."""
    atoms = Atoms(
        numbers=[1, 1, 1, 1],
        positions=[
            [0.2, 0.1, 0.0],
            [2.4, 0.3, 0.2],
            [0.4, 2.6, 0.1],
            [2.2, 2.1, 2.5],
        ],
        cell=[5.5, 5.5, 5.5],
        pbc=True,
    )
    return atoms


def pt7(side: float = 2.7, z: int = 13, jitter: float = 0.0, seed: int = 0) -> Atoms:
    """Pentagonal bipyramid. Atomic number 13 is aluminium for EAM-Al."""
    radius = side / (2.0 * math.sin(math.pi / 5.0))
    height = math.sqrt(side * side - radius * radius)
    positions = []
    for index in range(5):
        angle = 2.0 * math.pi * index / 5.0
        positions.append(
            [radius * math.cos(angle), radius * math.sin(angle), 0.0]
        )
    positions.append([0.0, 0.0, height])
    positions.append([0.0, 0.0, -height])
    coords = np.asarray(positions, dtype=np.float64)
    if jitter:
        rng = np.random.default_rng(seed)
        coords = coords + rng.normal(scale=jitter, size=coords.shape)
    return Atoms(
        numbers=[z] * 7,
        positions=coords,
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )


def raw_bytes(energy: float, forces: np.ndarray) -> bytes:
    """SHA input: one float64 energy followed by the force array bytes."""
    energy_bytes = np.array([energy], dtype=np.float64).tobytes()
    force_bytes = np.ascontiguousarray(forces, dtype=np.float64).tobytes()
    return energy_bytes + force_bytes
