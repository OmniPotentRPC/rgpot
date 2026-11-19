# MIT License
# Copyright 2023--present rgpot developers
"""Geometry builders and the reference-data text format shared by the QSC
reference generators and the pytest checks.

Data file grammar (``#`` starts a comment, one record per line):

    param Z n m epsilon c a
    case NAME
    periodic 0|1
    cutoff R
    cell ax ay az bx by bz cx cy cz      (rows are the lattice vectors)
    atom Z x y z fx fy fz
    energy E
    stress sxx syy szz syz sxz sxy       (optional, periodic cases)
    end
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Case:
    name: str
    periodic: bool
    cutoff: float
    cell: list  # 3x3, rows are lattice vectors
    species: list
    pos: list  # N x 3 floats
    energy: object = None
    forces: list = field(default_factory=list)  # N x 3
    stress: list | None = None  # Voigt xx yy zz yz xz xy


def cube(length):
    return [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]


FCC_BASIS = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))


def fcc_supercell(cells, lattice):
    pos = []
    for ix in range(cells):
        for iy in range(cells):
            for iz in range(cells):
                for bx, by, bz in FCC_BASIS:
                    pos.append(
                        [(ix + bx) * lattice, (iy + by) * lattice, (iz + bz) * lattice]
                    )
    return pos


def cuboctahedron(lattice, extra=()):
    """Centre atom, its 12 fcc neighbours, plus optional extra lattice sites."""
    h = lattice / 2.0
    pos = [[0.0, 0.0, 0.0]]
    for a, b in ((h, h), (h, -h), (-h, h), (-h, -h)):
        pos.append([a, b, 0.0])
        pos.append([a, 0.0, b])
        pos.append([0.0, a, b])
    for site in extra:
        pos.append([site[0] * lattice, site[1] * lattice, site[2] * lattice])
    return pos


def jitter(pos, amplitude, seed):
    rng = random.Random(seed)
    return [[x + amplitude * (2.0 * rng.random() - 1.0) for x in p] for p in pos]


def fmt_float(x):
    return repr(float(x))


def fmt_dec(x):
    if isinstance(x, Decimal):
        return format(x, ".20e")
    return repr(float(x))


def write_cases(path, header_lines, params, cases):
    with open(path, "w", encoding="ascii") as out:
        for line in header_lines:
            out.write(("# " + line).rstrip() + "\n")
        for z in sorted(params):
            out.write("param " + str(z) + " " + " ".join(fmt_float(v) for v in params[z]) + "\n")
        for case in cases:
            out.write(f"case {case.name}\n")
            out.write(f"periodic {int(case.periodic)}\n")
            out.write(f"cutoff {fmt_float(case.cutoff)}\n")
            out.write("cell " + " ".join(fmt_float(v) for row in case.cell for v in row) + "\n")
            for z, p, f in zip(case.species, case.pos, case.forces):
                out.write(
                    f"atom {z} "
                    + " ".join(fmt_float(v) for v in p)
                    + " "
                    + " ".join(fmt_dec(v) for v in f)
                    + "\n"
                )
            out.write(f"energy {fmt_dec(case.energy)}\n")
            if case.stress is not None:
                out.write("stress " + " ".join(fmt_dec(v) for v in case.stress) + "\n")
            out.write("end\n")


def read_cases(path):
    params = {}
    cases = []
    cur = None
    with open(path, encoding="ascii") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            key, *rest = line.split()
            if key == "param":
                params[int(rest[0])] = tuple(float(v) for v in rest[1:])
            elif key == "case":
                cur = Case(rest[0], False, 0.0, [], [], [])
            elif key == "periodic":
                cur.periodic = rest[0] == "1"
            elif key == "cutoff":
                cur.cutoff = float(rest[0])
            elif key == "cell":
                vals = [float(v) for v in rest]
                cur.cell = [vals[0:3], vals[3:6], vals[6:9]]
            elif key == "atom":
                cur.species.append(int(rest[0]))
                cur.pos.append([float(v) for v in rest[1:4]])
                cur.forces.append([Decimal(v) for v in rest[4:7]])
            elif key == "energy":
                cur.energy = Decimal(rest[0])
            elif key == "stress":
                cur.stress = [Decimal(v) for v in rest]
            elif key == "end":
                cases.append(cur)
                cur = None
    return params, cases
