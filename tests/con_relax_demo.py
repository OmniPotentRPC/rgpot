#!/usr/bin/env python3
"""readcon -> rgpot -> ASE: relax a structure loaded from an EON ``.con`` file.

End-to-end ergonomic pipeline:

1. ``readcon`` (readcon-core) loads an EON ``.con`` file straight into an
   :class:`ase.Atoms` via :func:`ase_helpers.atoms_from_con`.
2. :class:`RgpotCalculator` attaches an rgpot potential (here CuH2 over RPC).
3. A standard ASE optimizer relaxes the geometry.
4. ``atoms_to_con`` writes the relaxed atoms with precision omitted and
   canonical left off.

The reference ``.con`` is already at a CuH2 minimum, so we rattle it and show
that an ASE local relaxation recovers the original minimum energy -- a faithful
round-trip through all three components.

Run: ``RGPOT_POTSERV_BIN=<potserv> CON_FILE=<file.con> python tests/con_relax_demo.py``
(``CON_FILE`` defaults to readcon-core's bundled ``tiny_cuh2.con`` when present.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from ase.optimize import LBFGS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CppCore" / "rgpot" / "rpc"))
from ase_calculator import RgpotCalculator  # noqa: E402
from ase_helpers import atoms_from_con, atoms_to_con  # noqa: E402

_DEFAULT_CON = Path.home() / "Git/Github/Rust/readcon-core/resources/test/tiny_cuh2.con"


def main() -> int:
    server_bin = os.environ.get("RGPOT_POTSERV_BIN")
    if not server_bin:
        print("set RGPOT_POTSERV_BIN to the potserv binary", file=sys.stderr)
        return 2
    con_file = os.environ.get("CON_FILE", str(_DEFAULT_CON))
    if not Path(con_file).exists():
        print(f"con file not found: {con_file} (set CON_FILE)", file=sys.stderr)
        return 2

    with RgpotCalculator.spawn(server_bin=server_bin, potential="CuH2") as calc:
        ref = atoms_from_con(con_file, calc=calc)
        e_ref = ref.get_potential_energy()
        symbols = ref.get_chemical_symbols()

        # Rattle the loaded structure, then relax it back with ASE.
        work = ref.copy()
        work.calc = calc
        work.rattle(stdev=0.15, seed=1)
        e_rattled = work.get_potential_energy()

        opt = LBFGS(work, logfile=None)
        opt.run(fmax=0.02, steps=300)
        e_relaxed = work.get_potential_energy()
        fmax = float(np.abs(work.get_forces()).max())
        relaxed_con = Path("relaxed.con")
        atoms_to_con(relaxed_con, work)

    print("readcon -> rgpot -> ASE relaxation of CuH2")
    print(f"  loaded {Path(con_file).name}: {symbols}")
    print(f"  reference (.con) energy : {e_ref:.6f} eV")
    print(f"  after rattle(0.15)      : {e_rattled:.6f} eV")
    print(f"  after ASE LBFGS relax   : {e_relaxed:.6f} eV   (|Fmax| = {fmax:.4f})")
    print(f"  relaxed - reference     : {e_relaxed - e_ref:+.6f} eV")
    print(f"  wrote {relaxed_con.resolve()}")

    recovered = abs(e_relaxed - e_ref) < 1e-2 and e_relaxed <= e_rattled + 1e-6
    print("  RESULT:", "PASS -- ASE recovered the .con minimum through the rgpot calculator"
          if recovered else "FAIL")
    return 0 if recovered else 1


if __name__ == "__main__":
    raise SystemExit(main())
