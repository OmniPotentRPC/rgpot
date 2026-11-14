"""All-images Lennard-Jones and Morse against ASE and a replicated cell."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc100
from ase.calculators.fd import calculate_numerical_forces, calculate_numerical_stress
from ase.calculators.lj import LennardJones

import rgpot
from rgpot.ase import RgpotCalculator

RC = 2.5


def _primitive() -> Atoms:
    atoms = bulk("Ar", "fcc", a=1.56)
    atoms.set_cell(atoms.cell.array @ np.array([[1.01, 0.02, 0.0], [0.0, 0.99, 0.03], [0.0, 0.0, 1.02]]),
                   scale_atoms=True)
    return atoms


def _triclinic() -> Atoms:
    cell = 1.7 * np.array([[1.0, 0.0, 0.0], [0.35, 0.9, 0.0], [-0.2, 0.3, 1.1]])
    atoms = Atoms("Ar3", scaled_positions=[[0.1, 0.2, 0.3], [0.6, 0.55, 0.4], [0.35, 0.8, 0.85]],
                  cell=cell, pbc=True)
    atoms.positions += np.random.default_rng(7).uniform(-0.05, 0.05, (3, 3))
    return atoms


def _slab() -> Atoms:
    atoms = fcc100("Ar", size=(1, 1, 3), a=1.56, vacuum=RC + 0.5, periodic=True)
    atoms.positions += np.random.default_rng(8).uniform(-0.05, 0.05, atoms.positions.shape)
    return atoms


CELLS = {"primitive": _primitive, "triclinic": _triclinic, "slab": _slab}


@pytest.mark.parametrize("name", sorted(CELLS))
def test_lj_all_images_matches_ase(name: str) -> None:
    atoms = CELLS[name]()
    ours = atoms.copy()
    ours.calc = RgpotCalculator("lj", u0=1.0, psi=1.0, cutoff=RC, all_images=True)
    ref = atoms.copy()
    ref.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=RC, smooth=False)
    assert ours.get_potential_energy() == pytest.approx(ref.get_potential_energy(), abs=1e-10)
    np.testing.assert_allclose(ours.get_forces(), ref.get_forces(), atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(ours.get_stress(), ref.get_stress(), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_lj_minimum_image_refuses_narrow_cells(name: str) -> None:
    atoms = CELLS[name]()
    atoms.calc = RgpotCalculator("lj", cutoff=RC)
    with pytest.raises(ValueError, match="all-images mode"):
        atoms.get_potential_energy()


@pytest.mark.parametrize("name", sorted(CELLS))
def test_lj_all_images_matches_finite_differences(name: str) -> None:
    atoms = CELLS[name]()
    atoms.calc = RgpotCalculator("lj", cutoff=RC, all_images=True)
    np.testing.assert_allclose(
        atoms.get_forces(), calculate_numerical_forces(atoms, eps=1e-5), atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(
        atoms.get_stress(),
        calculate_numerical_stress(atoms, eps=1e-6, voigt=True),
        atol=1e-6,
        rtol=0.0,
    )


def test_morse_all_images_matches_a_supercell() -> None:
    atoms = bulk("Pt", "fcc", a=3.92)
    atoms.positions += [[0.1, -0.05, 0.02]]
    cutoff = 6.0
    pot = rgpot.MorsePot(cutoff=cutoff, all_images=True)
    assert pot.config()["all_images"] is True
    assert rgpot.MorsePot(cutoff=cutoff).config()["all_images"] is False
    small = pot.evaluate(atoms.positions, atoms.numbers, atoms.cell.array)
    big = atoms.repeat(6)
    folded = rgpot.MorsePot(cutoff=cutoff).evaluate(big.positions, big.numbers, big.cell.array)
    assert small["energy"] == pytest.approx(folded["energy"] / len(big), abs=1e-11)
    np.testing.assert_allclose(small["forces"], folded["forces"][:1], atol=1e-11, rtol=0.0)
