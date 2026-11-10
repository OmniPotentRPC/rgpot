#!/usr/bin/env python3
# Copyright (c) rgpot developers
"""Score an AOTI package against fairchem and a finite difference.

The geometries are small walks of s-tetrazine that stay inside the
cutoff, so the neighbour list does not change. fairchem builds its
own graph on each call. The finite difference is minus dE/dx.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_uma_aoti as exp
from uma_engine import UmaEngine

# Planar s-tetrazine, angstrom, centred in a 40 angstrom cell.
_TETRAZINE = [
    [12.5, 13.83, 12.5],
    [12.5, 11.17, 12.5],
    [13.65, 13.165, 12.5],
    [13.65, 11.835, 12.5],
    [11.35, 13.165, 12.5],
    [11.35, 11.835, 12.5],
    [12.5, 14.91, 12.5],
    [12.5, 10.09, 12.5],
]
_Z = [6, 6, 7, 7, 7, 7, 1, 1]
_BOX = [40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]


def _atoms(pos: np.ndarray) -> Atoms:
    atoms = Atoms(numbers=_Z, positions=pos, cell=np.array(_BOX).reshape(3, 3), pbc=True)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def _flat(pos: np.ndarray) -> list[float]:
    return [float(v) for v in np.asarray(pos, dtype=np.float64).reshape(-1)]


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: uma_off_reference.py LIB MODEL SCHEMA", file=sys.stderr)
        return 2
    library, model, schema = sys.argv[1:]
    base = np.array(_TETRAZINE, dtype=np.float64)
    # 0.2 angstrom walks. The molecule is about 5 angstroms across and
    # the cutoff is 6, so every pair stays a neighbour.
    walks = [base]
    for axis in range(3):
        walked = base.copy()
        walked[0, axis] += 0.2
        walks.append(walked)

    engine = UmaEngine(library, model, Path(schema))
    e_ref0, f_ref0, _pred = exp.ase_reference(
        _atoms(base), "uma-s-1p1", "cpu", "omol", "float64"
    )
    worst_f = 0.0
    worst_e = 0.0
    try:
        for index, pos in enumerate(walks):
            energy, forces = engine.force(_flat(pos), _Z, _BOX)
            e_ref, f_ref, _ = exp.ase_reference(
                _atoms(pos), "uma-s-1p1", "cpu", "omol", "float64"
            )
            de = abs(energy - e_ref)
            df = float(np.max(np.abs(np.array(forces).reshape(-1, 3) - f_ref)))
            worst_e = max(worst_e, de)
            worst_f = max(worst_f, df)
            print(
                f"walk {index}: dE={de:.3e} max|dF|={df:.3e} E={energy:.17g}",
                flush=True,
            )
            if de > 1e-4 or df > 1e-4:
                print("FAIL: AOTI does not match fairchem", file=sys.stderr)
                return 3
        step = 1e-3
        e0, f0 = engine.force(_flat(base), _Z, _BOX)
        # Central difference. A one-sided step of 1e-3 on this bond
        # leaves a 0.04 eV/Å quadratic remainder.
        component = int(np.argmax(np.abs(f0)))
        plus = base.copy()
        minus = base.copy()
        plus.reshape(-1)[component] += step
        minus.reshape(-1)[component] -= step
        e_plus, _ = engine.force(_flat(plus), _Z, _BOX)
        e_minus, _ = engine.force(_flat(minus), _Z, _BOX)
        finite = (e_minus - e_plus) / (2.0 * step)
        gap = abs(finite - f0[component])
        print(
            f"finite-difference: component={component} -dE/dx={finite:.8g} "
            f"force={f0[component]:.8g} gap={gap:.3e}"
        )
        if gap > 1e-3:
            print("FAIL: finite difference does not match the force", file=sys.stderr)
            return 4
    finally:
        engine.close()
    print(f"worst dE={worst_e:.3e} worst max|dF|={worst_f:.3e} ref0={e_ref0:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
