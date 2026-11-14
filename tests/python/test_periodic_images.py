"""Thin and sheared cells give the same energy per atom as their supercells.

A cell narrower than twice the cutoff sums every periodic image, and a
wide sheared cell is binned by its face spacing, so the energy per atom
and the forces agree with a supercell of the same crystal.
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


def _lower_triangular(atoms):
    atoms = atoms.copy()
    cell, _ = atoms.cell.standard_form()
    atoms.set_cell(cell, scale_atoms=True)
    return atoms


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
    "kind",
    [
        pytest.param("sw", marks=needs_fortran),
        pytest.param("tersoff", marks=needs_fortran),
        "stillinger_weber_si",
        "tersoff_sic",
    ],
)
def test_diamond_primitive_supercells_match_the_cubic_cell(kind) -> None:
    calc = RgpotCalculator(kind)
    cubic, _ = _per_atom(bulk("Si", "diamond", a=_SI, cubic=True).repeat(3), calc)
    prim = bulk("Si", "diamond", a=_SI)
    for prim in (prim, _lower_triangular(prim)):
        for n in range(1, 8):
            energy, _ = _per_atom(prim.repeat(n), calc)
            assert energy == pytest.approx(cubic, abs=1e-10), (prim.cell[:], n)


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


def _energies(atoms, kind, **params):
    """Per-atom energies of ``atoms``, checked to sum to its energy."""
    atoms = atoms.copy()
    atoms.calc = RgpotCalculator(kind, **params)
    energies = atoms.get_potential_energies()
    energy = atoms.get_potential_energy()
    assert abs(float(np.sum(energies)) - energy) <= 1e-10 * max(1.0, abs(energy))
    return energies


def _unwrapped(atoms):
    """Each atom moved by its own lattice vector, up to three cells away."""
    atoms = atoms.copy()
    index = np.arange(len(atoms))
    shifts = np.stack([index % 3 - 1, index % 7 - 3, (index % 2) * 2], axis=1)
    atoms.positions += shifts @ atoms.cell[:]
    return atoms


def _wrapped(atoms):
    atoms = atoms.copy()
    atoms.wrap()
    return atoms


def _same_shares(atoms, kind, **params):
    """The shares follow the atoms in lower-triangular form, wrapped and
    unwrapped."""
    ref = _energies(atoms, kind, **params)
    for other in (
        _lower_triangular(atoms),
        _wrapped(atoms),
        _unwrapped(_wrapped(atoms)),
        _unwrapped(_lower_triangular(atoms)),
    ):
        np.testing.assert_allclose(
            _energies(other, kind, **params), ref, atol=1e-10, rtol=0.0
        )
    return ref


_ENERGY_KINDS = [
    pytest.param("stillinger_weber_si", id="stillinger_weber_si"),
    pytest.param("tersoff_sic", id="tersoff_sic"),
]


@pytest.mark.parametrize("kind", _ENERGY_KINDS)
def test_diamond_primitive_supercell_per_atom_energies(kind) -> None:
    """Up to 4x4x4 the image table, from 5x5x5 the face-spacing bins."""
    cubic = bulk("Si", "diamond", a=_SI, cubic=True).repeat(3)
    site = float(np.mean(_energies(cubic, kind)))
    perfect = bulk("Si", "diamond", a=_SI)
    np.testing.assert_allclose(_energies(perfect, kind), site, atol=1e-10, rtol=0.0)
    prim = perfect.copy()
    prim.rattle(0.05, seed=5)
    one = _energies(prim, kind)
    for n in range(1, 8):
        shares = _same_shares(prim.repeat(n), kind).reshape(-1, len(prim))
        np.testing.assert_allclose(
            shares, np.broadcast_to(one, shares.shape), atol=1e-10, rtol=0.0
        )


_THIN = [
    pytest.param("lj", bulk("Ar", "fcc", a=_LJ_FCC), {"cutoff": 3.0}, id="lj"),
    pytest.param("morse", bulk("Pt", "fcc", a=3.92), {"cutoff": 6.0}, id="morse"),
    pytest.param(
        "eam_al", bulk("Al", "fcc", a=4.05), {}, id="eam_al", marks=needs_fortran
    ),
    pytest.param(
        "cuh2", bulk("Cu", "fcc", a=3.61), {}, id="cuh2", marks=needs_fortran
    ),
    pytest.param("stillinger_weber_si", bulk("Si", "diamond", a=_SI), {}, id="sw"),
    pytest.param("tersoff_sic", bulk("Si", "diamond", a=_SI), {}, id="tersoff"),
]


@pytest.mark.parametrize(("kind", "cell", "params"), _THIN)
def test_thin_cell_per_atom_energies_follow_the_cell_form(kind, cell, params) -> None:
    thin = cell.repeat((2, 1, 1))
    thin.rattle(0.05, seed=9)
    one = _same_shares(thin, kind, **params)
    big = _energies(thin.repeat(4), kind, **params).reshape(-1, len(thin))
    np.testing.assert_allclose(big, np.broadcast_to(one, big.shape), atol=1e-10, rtol=0.0)


@pytest.mark.parametrize(("kind", "cell", "params"), _THIN)
def test_slab_per_atom_energies(kind, cell, params) -> None:
    """A slab periodic in-plane with vacuum across, the (T, T, F) case.

    The calculator takes only fully periodic flags, so the open axis is
    vacuum wider than twice any search. The in-plane rows are sheared.
    """
    slab = cell.repeat((1, 1, 4))
    normal = np.cross(slab.cell[0], slab.cell[1])
    normal *= np.sign(normal @ slab.cell[2]) / np.linalg.norm(normal)
    slab.cell[2] = slab.cell[2] + 20.0 * normal
    slab.rattle(0.04, seed=11)
    one = _same_shares(slab, kind, **params)
    assert np.ptp(one) > 1e-3 * abs(float(one.min()))
    big = _energies(slab.repeat((3, 3, 1)), kind, **params).reshape(-1, len(slab))
    np.testing.assert_allclose(big, np.broadcast_to(one, big.shape), atol=1e-10, rtol=0.0)
    with pytest.raises(ValueError, match="non-periodic"):
        mixed = slab.copy()
        mixed.pbc = (True, True, False)
        _energies(mixed, kind, **params)


@pytest.mark.parametrize(
    ("kind", "cell", "params"),
    [
        pytest.param(
            "lj", bulk("Ar", "fcc", a=_LJ_FCC, cubic=True).repeat(12), {"cutoff": 3.0},
            id="lj",
        ),
        pytest.param(
            "eam_al", bulk("Al", "fcc", a=4.05, cubic=True).repeat(6), {},
            id="eam_al", marks=needs_fortran,
        ),
        pytest.param(
            "cuh2", bulk("Cu", "fcc", a=3.61, cubic=True).repeat(6), {},
            id="cuh2", marks=needs_fortran,
        ),
    ],
)
def test_atoms_outside_a_wide_cell_keep_their_shares(kind, cell, params) -> None:
    atoms = cell.copy()
    atoms.rattle(0.05, seed=13)
    ref = _energies(atoms, kind, **params)
    np.testing.assert_allclose(
        _energies(_unwrapped(atoms), kind, **params), ref, atol=1e-10, rtol=0.0
    )
    sheared = atoms.copy()
    sheared.set_cell(np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]) @ atoms.cell[:])
    np.testing.assert_allclose(
        _same_shares(sheared, kind, **params), ref, atol=1e-10, rtol=0.0
    )
