#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Generate Quantum Sutton-Chen mixture reference data from the oracle.

The oracle is ``qsc_oracle.py``: 50-digit ``decimal`` arithmetic on the
equations of Qi, Cagin, Kimura and Goddard, Phys. Rev. B 59, 3527 (1999).
It shares no code with rgpot and is not eOn. Run from the repository root::

    python3 scripts/qsc_reference/gen_oracle_mixtures.py \\
        --out CppCore/tests/data/qsc/oracle_mixtures.dat
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from decimal import Decimal as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qsc_common as common  # noqa: E402
import qsc_oracle as oracle  # noqa: E402

CLUSTER_CUTOFF = 8.0
DIMER_CUTOFF = 10.0
ORTHO_CUTOFF = 5.2
TRICLINIC_CUTOFF = 4.8
LATTICE = 3.8
ALL_METALS = (28, 29, 45, 46, 47, 77, 78, 79)

DIMERS = (
    ((29, 47), 2.65),
    ((29, 47), 3.10),
    ((29, 28), 2.40),
    ((46, 78), 2.70),
    ((47, 79), 2.90),
    ((45, 77), 2.60),
)

TRICLINIC = (
    (11.4, 0.0, 0.0),
    (0.8, 11.5, 0.0),
    (-0.5, 0.6, 11.3),
)


def build_cases():
    cases = []
    for (za, zb), r in DIMERS:
        name = f"{oracle.SYMBOLS[za]}{oracle.SYMBOLS[zb]}_dimer_{r:.2f}"
        cases.append(
            common.Case(
                name, False, DIMER_CUTOFF, common.cube(1000.0),
                [za, zb], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
            )
        )

    sites = ((1.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.0, 1.0, 0.0),
             (0.5, 1.0, 0.5), (0.0, 0.0, 1.0), (0.5, 0.5, 1.0))
    cluster = common.jitter(common.cuboctahedron(LATTICE, extra=sites), 0.08, 3001)
    species = [ALL_METALS[i % len(ALL_METALS)] for i in range(len(cluster))]
    cases.append(
        common.Case("eight_metal_cluster19", False, CLUSTER_CUTOFF,
                    common.cube(1000.0), species, cluster)
    )

    pos = common.jitter(common.fcc_supercell(3, LATTICE), 0.10, 3002)
    cycle = (29, 47, 28, 79)
    cases.append(
        common.Case("CuAgNiAu_ortho108", True, ORTHO_CUTOFF,
                    common.cube(3 * LATTICE), [cycle[i % 4] for i in range(len(pos))], pos)
    )

    cell = [list(row) for row in TRICLINIC]
    frac = [[p[k] / (3 * LATTICE) for k in range(3)] for p in common.fcc_supercell(3, LATTICE)]
    tri = [
        [sum(f[r] * cell[r][k] for r in range(3)) for k in range(3)] for f in frac
    ]
    tri = common.jitter(tri, 0.08, 3003)
    cycle = (29, 47, 46, 78)
    cases.append(
        common.Case("CuAgPdPt_triclinic108", True, TRICLINIC_CUTOFF, cell,
                    [cycle[i % 4] for i in range(len(tri))], tri)
    )
    return cases


def fill(case):
    pos = [[D(x) for x in p] for p in case.pos]
    cell = [[D(x) for x in row] for row in case.cell]
    energy, forces, stress = oracle.evaluate(
        case.species, pos, cell, case.periodic, case.cutoff, stress=case.periodic
    )
    case.energy = energy
    case.forces = forces
    case.stress = stress


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if not oracle.selfcheck(verbose=False):
        sys.exit("oracle finite-difference self check failed")
    cases = build_cases()
    for case in cases:
        fill(case)

    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256(open(os.path.join(here, "qsc_oracle.py"), "rb").read()).hexdigest()
    header = [
        "MIT License",
        "Copyright 2023--present rgpot developers",
        "",
        "Quantum Sutton-Chen mixture reference energies (eV), forces (eV/Angstrom) and",
        "Cauchy stress (eV/Angstrom^3, xx yy zz yz xz xy) from an independent oracle:",
        "scripts/qsc_reference/qsc_oracle.py, 50-digit decimal arithmetic on Eqs. (1a)-(1c)",
        "and (2) of Qi, Cagin, Kimura and Goddard, Phys. Rev. B 59, 3527 (1999),",
        "doi:10.1103/PhysRevB.59.3527, with all periodic images within the cutoff summed",
        "explicitly. The analytic gradient was checked against 50-digit central",
        "differences of the energy and of the strained energy before writing.",
        f"qsc_oracle.py sha256 {digest}",
        "This is not eOn output: the pinned eOn QSC disagrees on mixtures (see",
        "docs/orgmode/reference/qsc.org).",
    ]
    common.write_cases(args.out, header, oracle.QSC_TABLE, cases)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
