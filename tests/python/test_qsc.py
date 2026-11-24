# MIT License
# Copyright 2023--present rgpot developers

"""Tests for the Python and ASE Quantum Sutton-Chen (QSCPot) bindings."""

from __future__ import annotations

import math

import numpy as np
import pytest

import rgpot
from rgpot.ase import RgpotCalculator, available_kinds

DIMER_ABS_TOL = 1.0e-13
ENERGY_SUM_TOL = 1.0e-12
VIRIAL_SUM_TOL = 1.0e-10
HVP_STEP = 1.0e-5
HVP_FLOOR = 1.0e-3
HVP_ABS_TOL = 1.0e-7


def test_qsc_cu_and_cu_ag_dimer_reference():
    pot = rgpot.QSCPot(cutoff=10.0, verlet_skin=0.5, periodic=False, stress=True)
    assert pot.config() == {
        "cutoff": 10.0,
        "verlet_skin": 0.5,
        "periodic": False,
        "stress": True,
    }
    caps = pot.caps()
    assert caps["periodic"] is False
    assert caps["stress"] is True
    assert pot.has_analytic_hvp() is True

    # Cu dimer at r = a_Cu = 3.6030 Angstrom: E = eps * (1 - 2*c).
    a_cu = 3.6030
    eps_cu = 0.0057921
    c_cu = 84.843
    pos = np.array([[0.0, 0.0, 0.0], [a_cu, 0.0, 0.0]], dtype=np.float64)
    box = np.eye(3, dtype=np.float64) * 30.0
    energy, forces, variance = pot(pos, [29, 29], box)
    assert variance == 0.0
    assert math.isclose(
        energy, eps_cu * (1.0 - 2.0 * c_cu), rel_tol=0.0, abs_tol=DIMER_ABS_TOL
    )
    want_fx = eps_cu * (5.0 * c_cu - 10.0) / a_cu
    assert math.isclose(forces[0, 0], want_fx, rel_tol=0.0, abs_tol=DIMER_ABS_TOL)
    assert math.isclose(forces[1, 0], -want_fx, rel_tol=0.0, abs_tol=DIMER_ABS_TOL)


def test_qsc_per_atom_batch_and_hvp():
    pot = rgpot.QSCPot(cutoff=8.0, verlet_skin=0.5, periodic=True, stress=True)
    pos = np.array(
        [[0.0, 0.0, 0.0], [2.55, 0.10, -0.05], [1.25, 2.30, 0.20]],
        dtype=np.float64,
    )
    types = [29, 47, 79]
    box = np.eye(3, dtype=np.float64) * 20.0
    out = pot.evaluate_per_atom(pos, types, box)
    assert np.isclose(out["energies"].sum(), out["energy"], atol=ENERGY_SUM_TOL)
    s = out["stress"]
    voigt = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    volume = float(np.linalg.det(box))
    assert np.allclose(
        out["virials"].sum(axis=0), -volume * voigt, atol=VIRIAL_SUM_TOL
    )

    # Analytic HVP vs central difference.
    v = np.array(
        [[0.2, -0.1, 0.05], [-0.15, 0.25, -0.1], [0.05, -0.15, 0.2]],
        dtype=np.float64,
    )
    hv_auto = pot.hvp(pos, types, box, v, scheme="auto")
    hv_central = pot.hvp(pos, types, box, v, step=HVP_STEP, scheme="central")
    assert float(np.linalg.norm(hv_auto)) > HVP_FLOOR
    assert np.allclose(hv_auto, hv_central, atol=HVP_ABS_TOL)
    assert pot.hvp_counts()["analytic"] == 1

    # Batch evaluation matches serial calls bit for bit.
    frames = [pos, pos + 0.03]
    rows = pot.evaluate_batch(frames, [types, types], [box, box], n_threads=2)
    for frame, row in zip(frames, rows):
        e_serial, f_serial, _ = pot(frame, types, box)
        assert row["energy"] == e_serial
        assert np.array_equal(row["forces"], f_serial)


def test_qsc_set_element_parameter_and_ase_calculator():
    assert "qsc" in available_kinds()
    from ase import Atoms

    atoms = Atoms(
        symbols="Cu2Ag2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.85, 1.85, 0.0],
            [1.85, 0.0, 1.85],
            [0.0, 1.85, 1.85],
        ],
        cell=np.eye(3) * 18.0,
        pbc=True,
    )
    atoms.calc = RgpotCalculator("qsc", cutoff=8.0, stress=True)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()
    assert math.isfinite(e) and e < 0.0
    assert f.shape == (4, 3)
    assert s.shape == (6,)

    # Coincident atoms raise NumericalError.
    pot = rgpot.QSCPot()
    with pytest.raises(rgpot.NumericalError):
        pot(np.zeros((2, 3), dtype=np.float64), [29, 29], np.eye(3) * 20.0)
