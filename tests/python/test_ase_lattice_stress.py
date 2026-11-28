# Copyright 2023--present rgpot developers

"""Stress when the cell does not span three dimensions.

ASE 3.29 counts nonzero lattice vectors as ``Atoms.cell.rank``.
``LennardJones`` and ``EMT`` still return energy and forces below rank 3,
and ``get_stress`` raises ``PropertyNotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from rgpot.ase import RgpotCalculator

ATOL = 1e-9

LJ_U0 = 0.0104
LJ_PSI = 3.40
LJ_CUTOFF = 8.5


def _lj_ours() -> RgpotCalculator:
    return RgpotCalculator("lj", u0=LJ_U0, psi=LJ_PSI, cutoff=LJ_CUTOFF)


def _lj_ase() -> LennardJones:
    return LennardJones(epsilon=LJ_U0, sigma=LJ_PSI, rc=LJ_CUTOFF, smooth=False)


def _pair(distance: float = 3.9) -> Atoms:
    """Argon pair with no cell. ``cell.rank`` is 0."""
    return Atoms(
        "Ar2",
        positions=[[0.0, 0.0, 0.0], [distance, 0.02, -0.01]],
    )


def _energy_forces_match(ours: Atoms, ref: Atoms) -> None:
    assert ours.get_potential_energy() == pytest.approx(
        ref.get_potential_energy(), abs=ATOL
    )
    np.testing.assert_allclose(ours.get_forces(), ref.get_forces(), atol=ATOL, rtol=0.0)
    assert ours.get_potential_energy(force_consistent=True) == pytest.approx(
        ref.get_potential_energy(force_consistent=True), abs=ATOL
    )
    np.testing.assert_allclose(
        ours.get_potential_energies(),
        ref.get_potential_energies(),
        atol=ATOL,
        rtol=0.0,
    )


def _stress_refused(atoms: Atoms, ref: Atoms) -> None:
    with pytest.raises(PropertyNotImplementedError):
        ref.get_stress()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stresses()
    # A refused stress request leaves the cached energy in place.
    assert atoms.get_potential_energy() == pytest.approx(
        ref.get_potential_energy(), abs=ATOL
    )
    assert "stress" not in atoms.calc.results
    assert "stresses" not in atoms.calc.results


def test_rank0_matches_ase_lj_and_refuses_stress() -> None:
    bare = _pair()
    assert bare.cell.rank == 0
    ours = bare.copy()
    ref = bare.copy()
    ours.calc = _lj_ours()
    ref.calc = _lj_ase()
    _energy_forces_match(ours, ref)
    _stress_refused(ours, ref)

    fresh = bare.copy()
    calc = _lj_ours()
    with pytest.raises(PropertyNotImplementedError):
        calc.calculate(fresh, properties=["energy", "forces", "stress"])
    assert isinstance(calc.results["energy"], float)
    assert calc.results["free_energy"] == pytest.approx(calc.results["energy"], abs=ATOL)
    assert calc.results["forces"].shape == (len(fresh), 3)
    assert "stress" not in calc.results


def test_rank1_wire_matches_ase_lj_and_refuses_stress() -> None:
    wire = _pair()
    wire.set_cell([[40.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    wire.pbc = [True, False, False]
    assert wire.cell.rank == 1
    ours = wire.copy()
    ref = wire.copy()
    ours.calc = _lj_ours()
    ref.calc = _lj_ase()
    _energy_forces_match(ours, ref)
    _stress_refused(ours, ref)


def test_rank2_slab_matches_ase_lj_and_refuses_stress() -> None:
    slab = _pair()
    slab.set_cell([[40.0, 0.0, 0.0], [0.4, 40.0, 0.0], [0.0, 0.0, 0.0]])
    slab.pbc = [True, True, False]
    assert slab.cell.rank == 2
    ours = slab.copy()
    ref = slab.copy()
    ours.calc = _lj_ours()
    ref.calc = _lj_ase()
    _energy_forces_match(ours, ref)
    _stress_refused(ours, ref)


def test_rank3_nonperiodic_stress_matches_ase_lj() -> None:
    """A boxed cluster has a stress. ASE strains that three-vector cell."""
    boxed = _pair()
    boxed.set_cell([20.0, 21.0, 22.0])
    boxed.pbc = False
    assert boxed.cell.rank == 3
    ours = boxed.copy()
    ref = boxed.copy()
    ours.calc = _lj_ours()
    ref.calc = _lj_ase()
    _energy_forces_match(ours, ref)
    np.testing.assert_allclose(ours.get_stress(), ref.get_stress(), atol=ATOL, rtol=0.0)
    np.testing.assert_allclose(ours.get_stresses(), ref.get_stresses(), atol=ATOL, rtol=0.0)
    assert ours.get_stress().shape == (6,)
    assert ours.get_stresses().shape == (len(boxed), 6)


def test_emt_refuses_stress_when_the_cell_has_no_lattice() -> None:
    atoms = Atoms("Cu2", positions=[[0.0, 0.0, 0.0], [2.55, 0.04, -0.02]])
    assert atoms.cell.rank == 0
    ours = atoms.copy()
    ref = atoms.copy()
    ours.calc = RgpotCalculator("emt", numerical_stress=True)
    ref.calc = EMT()
    assert np.isfinite(ours.get_potential_energy())
    assert np.isfinite(ref.get_potential_energy())
    with pytest.raises(PropertyNotImplementedError):
        ref.get_stress()
    with pytest.raises(PropertyNotImplementedError):
        ours.get_stress()


def test_calculate_many_refuses_stress_below_rank_3() -> None:
    """A batch uses the same rank gate as one calculate call."""
    bare = _pair()
    assert bare.cell.rank < 3
    calc = _lj_ours()
    rows = calc.calculate_many([bare])
    assert "stress" not in rows[0]
    ref = bare.copy()
    ref.calc = _lj_ase()
    assert rows[0]["energy"] == pytest.approx(ref.get_potential_energy(), abs=ATOL)
    np.testing.assert_allclose(rows[0]["forces"], ref.get_forces(), atol=ATOL, rtol=0.0)
    with pytest.raises(PropertyNotImplementedError):
        ref.get_stress()
    with pytest.raises(PropertyNotImplementedError):
        calc.calculate_many([bare], properties=["energy", "forces", "stress"])

    boxed = _pair()
    boxed.set_cell([20.0, 21.0, 22.0])
    boxed.pbc = False
    assert boxed.cell.rank == 3
    ref_box = boxed.copy()
    ref_box.calc = _lj_ase()
    boxed_rows = calc.calculate_many(
        [boxed], properties=["energy", "forces", "stress"]
    )
    assert boxed_rows[0]["stress"].shape == (6,)
    np.testing.assert_allclose(
        boxed_rows[0]["stress"], ref_box.get_stress(), atol=ATOL, rtol=0.0
    )
