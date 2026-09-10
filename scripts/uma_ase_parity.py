#!/usr/bin/env python3
"""ASE reference for Baker HCN. Same geometry as CppCore/tests/uma_ase_parity.cc.

Default model is the in-tree metatomic LJ fixture so the C++ UmaPot path can
be compared without a HuggingFace download. Pass --fairchem to use the
FAIRChemCalculator on uma-s-1p1 / omol instead.
"""

from __future__ import annotations

import argparse
import sys

from ase import Atoms


HCN = Atoms(
    numbers=[6, 7, 1],
    positions=[
        [12.49734736216627162, 12.49892801474515913, 12.54059929828148512],
        [12.50115413363106498, 12.50036504272228832, 11.38209979880783251],
        [12.50149850420264563, 12.50069809648255514, 13.61514544631068446],
    ],
    cell=[[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]],
    pbc=True,
)


def metatomic_calc(model: str):
    try:
        from metatomic.torch.ase_calculator import MetatomicCalculator
    except ImportError:
        from metatomic_ase import MetatomicCalculator  # type: ignore
    return MetatomicCalculator(model, device="cpu")


def fairchem_calc(model: str, task: str):
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit(model, device="cpu")
    return FAIRChemCalculator(predictor, task_name=task)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="CppCore/tests/data/lj38/lennard-jones.pt",
    )
    parser.add_argument("--fairchem", action="store_true")
    parser.add_argument("--task-name", default="omol")
    args = parser.parse_args()

    atoms = HCN.copy()
    atoms.info.update({"charge": 0, "spin": 1, "spin_multiplicity": 1})
    if args.fairchem:
        atoms.calc = fairchem_calc(args.model, args.task_name)
        backend = "fairchem"
    else:
        atoms.calc = metatomic_calc(args.model)
        backend = "ase-metatomic"

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    print(f"backend={backend}")
    print(f"model={args.model}")
    print(f"energy={energy:.17g}")
    for i, row in enumerate(forces):
        print(f"force {i} {row[0]:.17g} {row[1]:.17g} {row[2]:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
