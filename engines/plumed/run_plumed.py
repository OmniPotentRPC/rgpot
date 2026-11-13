#!/usr/bin/env python3
"""Drive the RGPOT collective variable in PLUMED's default units.

The xyz file is nanometres. ``--debug-forces`` checks the analytic
derivatives against a finite difference of the collective variable.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


XYZ = """2
2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0
Ar 0.0 0.0 0.0
Ar 0.15 0.0 0.0
"""


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_plumed.py rgpot-lj libRgpotPlumed.so")
    tool, plugin = sys.argv[1:]
    work = Path.cwd()
    (work / "dimer.xyz").write_text(XYZ)
    factor = float(subprocess.check_output([tool, "--kJ-per-eV"], text=True).strip())
    (work / "plumed.dat").write_text(
        "\n".join(
            [
                f"LOAD FILE={Path(plugin).resolve()}",
                f"ene: RGPOT ATOMS=1,2 U0={factor:.16e} CUTOFF=0.8 PSI=0.1",
                "BIASVALUE ARG=ene",
                "PRINT ARG=ene FILE=COLVAR FMT=%16.10f",
                "",
            ]
        )
    )
    subprocess.run(
        ["plumed", "driver", "--plumed", "plumed.dat", "--ixyz", "dimer.xyz",
         "--debug-forces", "1e-4"],
        cwd=work, check=True,
    )
    values = [
        line for line in (work / "COLVAR").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if not values:
        raise SystemExit("PLUMED wrote no energy")
    energy = float(values[-1].split()[-1]) / factor
    direct = subprocess.check_output(
        [tool],
        input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True,
    )
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(energy - ref) > 1e-8:
        raise SystemExit(f"plumed {energy} eV, rgpot {ref} eV")
    print(f"plumed energy {energy:.10f} eV")


if __name__ == "__main__":
    main()
