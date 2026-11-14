"""Thin cells give the same energy per atom as their supercells.

A cell narrower than twice the cutoff sums every periodic image, so the
energy per atom and the forces agree with a supercell of the same crystal.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.fd import calculate_numerical_stress

import rgpot
from rgpot.ase import RgpotCalculator

_LJ_FCC = 2.0 ** (1.0 / 6.0) * np.sqrt(2.0)
_SI = 5.431

needs_fortran = pytest.mark.skipif(
    not rgpot.has_fortran_pots, reason="no Fortran potentials"
)


def _per_atom(atoms, calc):
    atoms = atoms.copy()
    atoms.calc = calc
    return atoms.get_potential_energy() / len(atoms), atoms.get_forces()


def _same_as_supercell(kind, thin, repeat, **params):
    thin = thin.copy()
    thin.rattle(0.05, seed=3)
    e_thin, f_thin = _per_atom(thin, RgpotCalculator(kind, **params))
    e_big, f_big = _per_atom(thin.repeat(repeat), RgpotCalculator(kind, **params))
    assert e_thin == pytest.approx(e_big, abs=1e-10)
    np.testing.assert_allclose(f_thin, f_big[: len(thin)], atol=1e-9, rtol=0.0)


def test_one_atom_fcc_lennard_jones_sees_its_own_images() -> None:
    calc = RgpotCalculator("lj", cutoff=3.0)
    one, _ = _per_atom(bulk("Ar", "fcc", a=_LJ_FCC), calc)
    four, _ = _per_atom(bulk("Ar", "fcc", a=_LJ_FCC, cubic=True), calc)
    wide, _ = _per_atom(bulk("Ar", "fcc", a=_LJ_FCC, cubic=True).repeat(4), calc)
    assert one == pytest.approx(-7.762386540408, abs=1e-10)
    assert one == pytest.approx(wide, abs=1e-10)
    assert four == pytest.approx(wide, abs=1e-10)


def test_thin_lennard_jones_cell_matches_its_supercell() -> None:
    _same_as_supercell(
        "lj", bulk("Ar", "fcc", a=_LJ_FCC, cubic=True), (4, 4, 4), cutoff=3.0
    )


@needs_fortran
@pytest.mark.parametrize(
    ("kind", "cell"),
    [
        ("eam_al", bulk("Al", "fcc", a=4.05)),
        ("cuh2", bulk("Cu", "fcc", a=3.61)),
        ("fehe", bulk("Fe", "bcc", a=2.87)),
        ("sw", bulk("Si", "diamond", a=_SI)),
        ("tersoff", bulk("Si", "diamond", a=_SI)),
        ("edip", bulk("Si", "diamond", a=_SI)),
        ("lenosky", bulk("Si", "diamond", a=_SI)),
    ],
)
def test_thin_fortran_cell_matches_its_supercell(kind, cell) -> None:
    _same_as_supercell(kind, cell, (4, 4, 4))


@pytest.mark.parametrize(
    ("kind", "cell", "params"),
    [
        ("lj", bulk("Ar", "fcc", a=_LJ_FCC), {"cutoff": 3.0}),
        pytest.param("sw", bulk("Si", "diamond", a=_SI), {}, marks=needs_fortran),
    ],
)
def test_thin_cell_stress_matches_finite_difference(kind, cell, params) -> None:
    atoms = cell.copy()
    atoms.rattle(0.05, seed=3)
    atoms.calc = RgpotCalculator(kind, **params)
    numeric = calculate_numerical_stress(atoms, eps=1e-6)
    np.testing.assert_allclose(atoms.get_stress(), numeric, atol=1e-7, rtol=0.0)
