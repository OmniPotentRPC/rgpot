"""Tests for GuptaPot and the ASE calculator 'gupta' binding."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)

import rgpot
from rgpot.ase import RgpotCalculator, available_kinds


def _cluster13(r0: float = 2.884) -> np.ndarray:
    s = r0 / np.sqrt(2.0)
    raw = np.array(
        [
            [0.0, 0.0, 0.0],
            [s, s, 0.0],
            [s, -s, 0.0],
            [-s, s, 0.0],
            [-s, -s, 0.0],
            [s, 0.0, s],
            [s, 0.0, -s],
            [-s, 0.0, s],
            [-s, 0.0, -s],
            [0.0, s, s],
            [0.0, s, -s],
            [0.0, -s, s],
            [0.0, -s, -s],
        ],
        dtype=np.float64,
    )
    for i in range(13):
        for k in range(3):
            raw[i, k] += 0.03 * r0 * np.sin(1.7 * (3 * i + k) + 0.4)
    return raw


def _fcc_supercell(cells: int = 2, lattice: float = 4.08) -> Atoms:
    basis = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    cell = np.array(
        [
            [cells * lattice, 0.0, 0.0],
            [0.05 * cells * lattice, cells * lattice, 0.0],
            [0.03 * cells * lattice, -0.04 * cells * lattice, cells * lattice],
        ],
        dtype=np.float64,
    )
    frac = []
    idx = 0
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                for b in basis:
                    frac.append(
                        [
                            (ix + b[0]) / cells + 0.004 * np.sin(1.3 * (3 * idx) + 0.2),
                            (iy + b[1]) / cells + 0.004 * np.sin(1.3 * (3 * idx + 1) + 0.5),
                            (iz + b[2]) / cells + 0.004 * np.sin(1.3 * (3 * idx + 2) + 0.8),
                        ]
                    )
                    idx += 1
    pos = np.asarray(frac, dtype=np.float64) @ cell
    return Atoms(symbols=["X"] * len(pos), positions=pos, cell=cell, pbc=True)


def test_gupta_in_available_kinds() -> None:
    kinds = available_kinds()
    assert "gupta" in kinds
    assert "guptapot" in kinds


def test_gupta_dimer_at_r0() -> None:
    a, xi, p, q, r0 = 0.1477, 0.8900, 9.689, 4.602, 1.0
    pot = rgpot.GuptaPot(A=a, xi=xi, p=p, q=q, r0=r0)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    box = np.eye(3, dtype=np.float64) * 20.0
    energy, forces, variance = pot(pos, [1, 1], box)
    assert variance == 0.0
    assert energy == pytest.approx(2.0 * a - 2.0 * xi, abs=1e-12)
    assert forces.shape == (2, 3)


def test_gupta_ase_cluster_forces_and_invariance() -> None:
    params = dict(A=0.2061, xi=1.790, p=10.229, q=4.036, r0=2.884)
    pos = _cluster13(params["r0"])
    atoms = Atoms(symbols=["X"] * 13, positions=pos, pbc=False)
    atoms.calc = RgpotCalculator("gupta", **params)
    analytic = atoms.get_forces()
    numeric = calculate_numerical_forces(atoms, eps=1e-6)
    np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=0.0)

    e_ref = atoms.get_potential_energy()
    shifted = atoms.copy()
    shifted.positions += np.array([2.5, -1.7, 4.2])
    shifted.calc = RgpotCalculator("gupta", **params)
    assert shifted.get_potential_energy() == pytest.approx(e_ref, abs=1e-12)
    np.testing.assert_allclose(shifted.get_forces(), analytic, atol=1e-12, rtol=0.0)

    rotated = atoms.copy()
    theta, phi = 0.7, -0.5
    rz = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    rot = rz @ rx
    rotated.positions = atoms.positions @ rot.T
    rotated.calc = RgpotCalculator("gupta", **params)
    assert rotated.get_potential_energy() == pytest.approx(e_ref, abs=1e-12)
    np.testing.assert_allclose(rotated.get_forces(), analytic @ rot.T, atol=1e-12, rtol=0.0)


def test_gupta_ase_periodic_forces_and_stress() -> None:
    lattice = 4.08
    atoms = _fcc_supercell(cells=2, lattice=lattice)
    atoms.calc = RgpotCalculator(
        "gupta",
        A=0.2061,
        xi=1.790,
        p=10.229,
        q=4.036,
        r0=lattice / np.sqrt(2.0),
        cutoff=3.9,
        periodic=True,
        stress=True,
    )
    assert "stress" in atoms.calc.implemented_properties
    analytic_f = atoms.get_forces()
    numeric_f = calculate_numerical_forces(atoms, eps=1e-6)
    np.testing.assert_allclose(analytic_f, numeric_f, atol=1e-6, rtol=0.0)

    analytic_s = atoms.get_stress()
    numeric_s = calculate_numerical_stress(atoms, eps=1e-6, voigt=True)
    np.testing.assert_allclose(analytic_s, numeric_s, atol=1e-6, rtol=0.0)
