#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the reference runs for the tabulated EAM tests.

The runs use the Large-scale Atomic/Molecular Massively Parallel
Simulator (LAMMPS) on two potential files from its potentials/
directory, copied here unchanged:

  Cu_u3.eam       funcfl, pair_style eam, Foiles, Baskes and Daw,
                  Phys. Rev. B 33, 7983 (1986).
  CuNi.eam.alloy  setfl, pair_style eam/alloy, Onat and Durukanoglu,
                  J. Phys.: Condens. Matter 26, 035404 (2014).

Each case is a jittered fcc crystal, cubic or sheared, with the cutoff
under half the smallest face spacing, so one image of each partner is
the whole sum.
The CuNi case gives each site Ni or Cu at random.
Positions are written as LAMMPS stores them after read_data.
The stress is minus the LAMMPS virial pressure, in eV/Angstrom^3.

Usage: make_lammps_ref.py [output directory].
It needs the lammps Python module with the MANYBODY package.
"""

import pathlib
import sys
import tempfile

import lammps
import numpy as np

HERE = pathlib.Path(__file__).parent
# name, file, pair_style, elements (type order), cutoff, lattice, cells,
# sheared, seed
CASES = (
    ("cu_u3_cubic", "Cu_u3.eam", "eam", ("Cu",), 4.95, 3.615, 3, False, 21),
    ("cuni_sheared", "CuNi.eam.alloy", "eam/alloy", ("Ni", "Cu"), 6.394332378,
     3.56, 4, True, 22),
)
ZNUM = {"Cu": 29, "Ni": 28}


def crystal(lattice, cells, sheared, seed, cutoff, ntypes):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    grid = np.array(np.meshgrid(*[range(cells)] * 3, indexing="ij")).reshape(3, -1).T
    frac = (grid[:, None, :] + basis[None]).reshape(-1, 3) / cells
    box = np.diag([cells * lattice] * 3)
    if sheared:
        box[1, 0], box[2, 0], box[2, 1] = 1.0, 0.5, -0.8
    rng = np.random.default_rng(seed)
    pos = frac @ box + rng.uniform(-0.12, 0.12, frac.shape)
    pos = np.mod(pos @ np.linalg.inv(box), 1.0) @ box
    types = rng.integers(1, ntypes + 1, len(pos))
    for k in range(3):
        normal = np.cross(box[(k + 1) % 3], box[(k + 2) % 3])
        assert abs(np.linalg.det(box)) / np.linalg.norm(normal) > 2 * cutoff
    return box, pos, types


def run_lammps(box, pos, types, potfile, style, elements):
    with tempfile.TemporaryDirectory() as tmp:
        data = pathlib.Path(tmp) / "case.data"
        data.write_text(
            f"lammps data\n\n{len(pos)} atoms\n{len(elements)} atom types\n"
            f"0 {box[0, 0]:.17g} xlo xhi\n0 {box[1, 1]:.17g} ylo yhi\n"
            f"0 {box[2, 2]:.17g} zlo zhi\n"
            f"{box[1, 0]:.17g} {box[2, 0]:.17g} {box[2, 1]:.17g} xy xz yz\n\n"
            # Masses do not enter a run 0.
            "Masses\n\n"
            + "".join(f"{k + 1} 1.0\n" for k in range(len(elements)))
            + "\nAtoms # atomic\n\n"
            + "".join(f"{i + 1} {t} {x:.17g} {y:.17g} {z:.17g}\n"
                      for i, (t, (x, y, z)) in enumerate(zip(types, pos)))
        )
        coeff = f"* * {HERE / potfile}"
        if style != "eam":
            coeff += " " + " ".join(elements)
        lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
        lmp.commands_list([
            "units metal", "atom_style atomic", "boundary p p p",
            "atom_modify map array", f"read_data {data}", f"pair_style {style}",
            f"pair_coeff {coeff}", "neighbor 1.0 bin",
            "compute vir all pressure NULL virial", "thermo_style custom pe",
            "run 0",
        ])
    nat = len(pos)
    x = np.array(lmp.gather_atoms("x", 1, 3)[: 3 * nat]).reshape(nat, 3)
    f = np.array(lmp.gather_atoms("f", 1, 3)[: 3 * nat]).reshape(nat, 3)
    t = np.array(lmp.gather_atoms("type", 0, 1)[:nat])
    p = lmp.extract_compute("vir", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR)
    # LAMMPS order is xx yy zz xy xz yz; the file order is xx yy zz yz xz xy.
    stress = -np.array([p[0], p[1], p[2], p[5], p[4], p[3]]) / lmp.extract_global("nktv2p")
    result = (lmp.get_thermo("pe"), x, f, t, stress, lmp.version())
    lmp.close()
    print("largest remap on read_data", np.abs(x - pos).max())
    return result


def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    for name, potfile, style, elements, cutoff, lattice, cells, sheared, seed in CASES:
        box, pos, types = crystal(lattice, cells, sheared, seed, cutoff, len(elements))
        energy, x, f, t, stress, version = run_lammps(box, pos, types, potfile, style, elements)
        znum = [ZNUM[elements[k - 1]] for k in t]
        rows = "".join(f"{z} " + " ".join(f"{v:.17g}" for v in (*xi, *fi)) + "\n"
                       for z, xi, fi in zip(znum, x, f))
        (outdir / f"{name}.ref").write_text(
            f"# LAMMPS {version} pair_style {style}, {potfile}\n"
            f"box {' '.join(f'{v:.17g}' for v in box.ravel())}\n"
            f"energy {energy:.17g}\nstress {' '.join(f'{v:.17g}' for v in stress)}\n"
            f"atoms {len(x)}\n{rows}"
        )
        print(name, f"energy {energy:.12f} stress", " ".join(f"{v:.6e}" for v in stress))


if __name__ == "__main__":
    main()
