#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the reference runs for the Sutton-Chen stress tests.

The runs use the Large-scale Atomic/Molecular Massively Parallel
Simulator (LAMMPS).
Sutton-Chen is a Finnis-Sinclair form, so LAMMPS runs it as pair_style
eam/fs from a table of the embedding -epsilon c sqrt(rho), the density
(a / r)^m and r times the pair term epsilon (a / r)^n.
Each case is a jittered 108-atom platinum crystal, cubic or sheared,
with the cutoff under half the smallest face spacing, so one image of
each partner is the whole sum.
Positions are written as LAMMPS stores them after read_data.
The stress is minus the LAMMPS virial pressure, in eV/Angstrom^3.

Usage: make_lammps_ref.py [table points] [output directory].
It needs the lammps Python module with the MANYBODY package.
"""

import pathlib
import sys
import tempfile

import lammps
import numpy as np

EPS, C, A, N, M, MASS = 1.98330e-2, 34.408, 3.92, 10, 8, 195.084
CUTOFF, RHO_MAX = 5.5, 1000.0


def case(sheared, seed):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array(np.meshgrid(*[range(3)] * 3, indexing="ij")).reshape(3, -1).T
    frac = (cells[:, None, :] + basis[None]).reshape(-1, 3) / 3
    box = np.diag([3 * A] * 3)
    if sheared:
        box[1, 0], box[2, 0], box[2, 1] = 1.0, 0.5, -0.8
    pos = frac @ box + np.random.default_rng(seed).uniform(-0.12, 0.12, frac.shape)
    pos = np.mod(pos @ np.linalg.inv(box), 1.0) @ box
    for k in range(3):
        normal = np.cross(box[(k + 1) % 3], box[(k + 2) % 3])
        assert abs(np.linalg.det(box)) / np.linalg.norm(normal) > 2 * CUTOFF
    return box, pos


def write_table(path, nr):
    dr, drho = CUTOFF / (nr - 1), RHO_MAX / (nr - 1)
    r = np.maximum(np.arange(nr) * dr, 1.0)
    rho = np.arange(nr) * drho
    columns = (-EPS * C * np.sqrt(rho), (A / r) ** M, r * EPS * (A / r) ** N)
    with open(path, "w") as out:
        out.write("Sutton-Chen platinum\nn 10 m 8\nmake_lammps_ref.py\n1 Pt\n")
        out.write(f"{nr} {drho:.17g} {nr} {dr:.17g} {CUTOFF:.17g}\n78 {MASS} {A} fcc\n")
        for column in columns:
            out.write("".join(f"{v:.17g}\n" for v in column))


def run_lammps(box, pos, tmp, nr):
    write_table(tmp / "sc.eam.fs", nr)
    (tmp / "sc.data").write_text(
        f"lammps data\n\n{len(pos)} atoms\n1 atom types\n"
        f"0 {box[0, 0]:.17g} xlo xhi\n0 {box[1, 1]:.17g} ylo yhi\n"
        f"0 {box[2, 2]:.17g} zlo zhi\n"
        f"{box[1, 0]:.17g} {box[2, 0]:.17g} {box[2, 1]:.17g} xy xz yz\n\n"
        f"Masses\n\n1 {MASS}\n\nAtoms # atomic\n\n"
        + "".join(f"{i + 1} 1 {x:.17g} {y:.17g} {z:.17g}\n" for i, (x, y, z) in enumerate(pos))
    )
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_list([
        "units metal", "atom_style atomic", "boundary p p p", "atom_modify map array",
        f"read_data {tmp / 'sc.data'}", "pair_style eam/fs",
        f"pair_coeff * * {tmp / 'sc.eam.fs'} Pt", "neighbor 1.0 bin",
        "compute vir all pressure NULL virial", "thermo_style custom pe", "run 0",
    ])
    nat = len(pos)
    x = np.array(lmp.gather_atoms("x", 1, 3)[: 3 * nat]).reshape(nat, 3)
    f = np.array(lmp.gather_atoms("f", 1, 3)[: 3 * nat]).reshape(nat, 3)
    p = lmp.extract_compute("vir", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR)
    # LAMMPS order is xx yy zz xy xz yz; the file order is xx yy zz yz xz xy.
    stress = -np.array([p[0], p[1], p[2], p[5], p[4], p[3]]) / lmp.extract_global("nktv2p")
    result = (lmp.get_thermo("pe"), x, f, stress, lmp.version())
    lmp.close()
    print("largest remap on read_data", np.abs(x - pos).max())
    return result


def main():
    nr = int(sys.argv[1]) if len(sys.argv) > 1 else 50001
    outdir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(__file__).parent
    for name, sheared, seed in (("pt_cubic", False, 11), ("pt_sheared", True, 12)):
        box, pos = case(sheared, seed)
        with tempfile.TemporaryDirectory() as tmp:
            energy, x, f, stress, version = run_lammps(box, pos, pathlib.Path(tmp), nr)
        rows = "".join(" ".join(f"{v:.17g}" for v in (*xi, *fi)) + "\n" for xi, fi in zip(x, f))
        (outdir / f"{name}.ref").write_text(
            f"# LAMMPS {version} pair_style eam/fs, table nr = nrho = {nr}\n"
            f"cutoff {CUTOFF:.17g}\nbox {' '.join(f'{v:.17g}' for v in box.ravel())}\n"
            f"energy {energy:.17g}\nstress {' '.join(f'{v:.17g}' for v in stress)}\n"
            f"atoms {len(x)}\n{rows}"
        )
        print(name, f"energy {energy:.12f} stress", " ".join(f"{v:.6e}" for v in stress))


if __name__ == "__main__":
    main()
