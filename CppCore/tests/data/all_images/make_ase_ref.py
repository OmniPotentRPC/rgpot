#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the ASE references for the all-images pair tests.

Every cell is narrower than twice the cutoff, so a pair has several
images inside it and an atom meets its own images. ASE sums all of them.

  lj_*   ase.calculators.lj.LennardJones, sigma 1, epsilon 1, rc 2.5,
         smooth=False: each pair term is shifted to zero at rc, the
         convention of rgpot::LJPot.
  eam_*  ase.calculators.eam.EAM on synthetic.eam.alloy, which this
         script also writes: two elements (Cu and Ni numbers) with
         smooth analytic functions on a fine grid, so the cubic
         interpolations of ASE and of rgpot::EAMTablePot agree closely.

Cases per potential: a one-atom fcc primitive cell under a small shear,
a three-atom triclinic cell, and a three-layer fcc(100) slab with a
one-atom lateral cell and vacuum wider than twice the cutoff.

Rows are written as the LAMMPS references of ../eam_table: box (rows),
energy, stress xx yy zz yz xz xy in ASE's sign (1/V) dE/dstrain, then
"atoms n" and one row per atom: Z x y z fx fy fz.

Usage: make_ase_ref.py [output directory]. It needs ase and numpy.
"""

import pathlib
import sys

import ase
import numpy as np
from ase import Atoms
from ase.calculators.eam import EAM
from ase.calculators.lj import LennardJones
from ase.symbols import string2symbols

HERE = pathlib.Path(__file__).parent
SETFL = "synthetic.eam.alloy"
LJ_RC = 2.5
EAM_RC = 5.0
NR = 1001
NRHO = 1001
DR = EAM_RC / (NR - 1)
RHO_MAX = 50.0
DRHO = RHO_MAX / (NRHO - 1)
# element: Z, mass, lattice, density f, beta, r_e, embedding E
ELEMENTS = {
    "Cu": (29, 63.546, 3.615, 1.0, 5.0, 2.556, 2.0),
    "Ni": (28, 58.693, 3.52, 1.2, 5.2, 2.49, 2.3),
}
# pair: D, alpha, r0
PAIRS = {
    ("Cu", "Cu"): (0.30, 1.40, 2.60),
    ("Ni", "Cu"): (0.32, 1.45, 2.55),
    ("Ni", "Ni"): (0.35, 1.50, 2.50),
}


def cutoff(r):
    x = (r - EAM_RC) / 1.0
    return np.where(r < EAM_RC, x**4 / (1.0 + x**4), 0.0)


def write_setfl(path):
    r = np.arange(NR) * DR
    rho = np.arange(NRHO) * DRHO
    names = ("Cu", "Ni")
    out = [
        "synthetic two-element EAM for the rgpot all-images tests",
        "smooth analytic functions, written by make_ase_ref.py",
        "not a fitted potential",
        f"2 {' '.join(names)}",
        f"{NRHO} {DRHO:.16e} {NR} {DR:.16e} {EAM_RC:.16e}",
    ]

    def block(values):
        for k in range(0, len(values), 5):
            out.append(" ".join(f"{v:.16e}" for v in values[k : k + 5]))

    for name in names:
        z, mass, lattice, f, beta, r_e, emb = ELEMENTS[name]
        out.append(f"{z} {mass} {lattice} fcc")
        block(-emb * np.sqrt(rho + 1.0) + emb)
        block(f * np.exp(-beta * (r / r_e - 1.0)) * cutoff(r))
    for i, a in enumerate(names):
        for b in names[: i + 1]:
            d, alpha, r0 = PAIRS[(a, b)]
            phi = d * (np.exp(-2 * alpha * (r - r0)) - 2 * np.exp(-alpha * (r - r0)))
            block(r * phi * cutoff(r))
    path.write_text("\n".join(out) + "\n")


def jitter(atoms, seed, size):
    rng = np.random.default_rng(seed)
    atoms.positions += rng.uniform(-size, size, atoms.positions.shape)
    return atoms


def primitive(a, symbol):
    cell = 0.5 * a * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], float)
    cell = cell @ np.array([[1.01, 0.02, -0.01], [0.0, 0.985, 0.03], [0.0, 0.0, 1.02]])
    # Off the origin and outside the cell: rgpot must not assume wrapped input.
    return Atoms(symbol, positions=[[0.3 * a, -0.2 * a, 1.1 * a]], cell=cell, pbc=True)


def triclinic(a, symbols, seed):
    cell = a * np.array([[1.0, 0.0, 0.0], [0.35, 0.9, 0.0], [-0.2, 0.3, 1.1]])
    frac = np.array([[0.1, 0.2, 0.3], [0.6, 0.55, 0.4], [0.35, 0.8, 0.85]])
    atoms = Atoms(symbols, scaled_positions=frac, cell=cell, pbc=True)
    return jitter(atoms, seed, 0.03 * a)


def slab(a, symbols, rc, seed):
    side = a / np.sqrt(2.0)
    layers = len(string2symbols(symbols))
    pos = [[0.5 * side * (k % 2), 0.5 * side * (k % 2), 0.5 * a * k] for k in range(layers)]
    height = 0.5 * a * (layers - 1) + 2.0 * rc + 1.0
    atoms = Atoms(symbols, positions=pos, cell=[side, side, height], pbc=True)
    return jitter(atoms, seed, 0.03 * a)


def write_ref(path, header, atoms):
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress(voigt=True)
    lines = [f"# {header}"]
    lines.append("box " + " ".join(repr(float(v)) for v in atoms.cell.array.reshape(-1)))
    lines.append(f"energy {float(e)!r}")
    lines.append("stress " + " ".join(repr(float(v)) for v in s))
    lines.append(f"atoms {len(atoms)}")
    for z, x, g in zip(atoms.numbers, atoms.positions, f):
        lines.append(" ".join([str(int(z))] + [repr(float(v)) for v in (*x, *g)]))
    path.write_text("\n".join(lines) + "\n")


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    write_setfl(out / SETFL)
    lj_a = 1.56
    cases = {
        "lj_primitive": primitive(lj_a, "Ar"),
        "lj_triclinic": triclinic(1.1 * lj_a, "Ar3", 31),
        "lj_slab": slab(lj_a, "Ar3", LJ_RC, 32),
    }
    for name, atoms in cases.items():
        atoms.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=LJ_RC, smooth=False)
        write_ref(out / f"{name}.ref",
                  f"ASE {ase.__version__} LennardJones sigma 1 epsilon 1 rc {LJ_RC} smooth=False",
                  atoms)
    cases = {
        "eam_primitive": primitive(3.615, "Cu"),
        "eam_triclinic": triclinic(1.05 * 3.6, "CuNiCu", 41),
        "eam_slab": slab(3.56, "NiCuNi", EAM_RC, 42),
    }
    for name, atoms in cases.items():
        atoms.calc = EAM(potential=str(out / SETFL))
        write_ref(out / f"{name}.ref", f"ASE {ase.__version__} EAM {SETFL}", atoms)


if __name__ == "__main__":
    main()
