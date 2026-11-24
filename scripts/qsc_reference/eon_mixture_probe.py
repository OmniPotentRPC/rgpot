#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Show where the pinned eOn QSC departs from the equations on mixtures.

Not part of any test and not a source of reference data. For the
orthorhombic cases of ``oracle_mixtures.dat`` it runs three evaluations and
reports the largest energy and force difference of each against the oracle:

* eOn as committed (QSC.cpp unmodified);
* eOn with its parameter caching removed (a patched copy of QSC.cpp in a
  temporary tree: the ``prev_i_atomic_number`` / ``prev_j_atomic_number``
  guards at lines 142, 153, 207 and 219 become ``if (true)``).

Each is compared with the oracle twice: with the paper's density term
(``(a_ij / r)^m_ij``, Eq. 1c) and with the density term eOn implements
(``(a_j / r)^m_j``, the neighbour's own parameters). Run like gen_eon_pure.py::

    (cd "$EON_SRC" && pixi run python \\
        "$RGPOT/scripts/qsc_reference/eon_mixture_probe.py" \\
        --eon-src "$EON_SRC" --eon-build "$EON_SRC/builddir" \\
        --data "$RGPOT/CppCore/tests/data/qsc/oracle_mixtures.dat")

with ``EON_SRC`` the pinned eOn checkout and ``RGPOT`` this repository.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from decimal import Decimal as D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_eon_pure as gen  # noqa: E402
import qsc_common as common  # noqa: E402
import qsc_oracle as oracle  # noqa: E402

GUARDS = (
    "if (prev_i_atomic_number != atomicNrs[i])",
    "if (prev_j_atomic_number != atomicNrs[j])",
)


def patched_tree(src):
    """A temporary tree whose QSC.cpp always re-fetches its parameters."""
    root = tempfile.mkdtemp(prefix="qsc-eon-nocache-")
    client = os.path.join(root, "client")
    os.makedirs(os.path.join(client, "potentials", "QSC"))
    for path in glob.glob(os.path.join(src, "client", "*")):
        if os.path.basename(path) == "potentials":
            continue
        os.symlink(path, os.path.join(client, os.path.basename(path)))
    qdir = os.path.join(client, "potentials", "QSC")
    for name in ("QSC.h", "Parameters.h"):
        os.symlink(os.path.join(src, "client/potentials/QSC", name), os.path.join(qdir, name))
    text = open(os.path.join(src, "client/potentials/QSC/QSC.cpp"), encoding="utf-8").read()
    for guard in GUARDS:
        assert guard in text
        text = text.replace(guard, "if (true)")
    cpp = os.path.join(qdir, "QSC.cpp")
    open(cpp, "w", encoding="utf-8").write(text)
    return root, cpp


def deviations(case, got_e, got_f, rule):
    pos = [[D(x) for x in p] for p in case.pos]
    cell = [[D(x) for x in r] for r in case.cell]
    e, f, _ = oracle.evaluate(case.species, pos, cell, case.periodic, case.cutoff, density=rule)
    de = abs(D(got_e) - e)
    df = max(abs(D(got_f[i][k]) - f[i][k]) for i in range(len(f)) for k in range(3))
    return float(de), float(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eon-src", required=True)
    ap.add_argument("--eon-build", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cxx", default="g++")
    args = ap.parse_args(argv)
    src = os.path.abspath(args.eon_src)
    build = os.path.abspath(args.eon_build)
    prefix = gen.toolchain_prefix(args.cxx)
    work = tempfile.mkdtemp(prefix="qsc-probe-")

    _, cases = common.read_cases(args.data)
    cases = [c for c in cases if c.name != "CuAgPdPt_triclinic108"]
    overrides = {29: oracle.QSC_TABLE[29]}

    root, cpp = patched_tree(src)
    variants = {
        "eOn as committed": gen.compile_driver(args.cxx, "-O2", src, build, prefix, work, "stock")[0],
        "eOn without parameter caching": gen.compile_driver(
            args.cxx, "-O2", root, build, prefix, work, "nocache", qsc_cpp=cpp
        )[0],
    }
    print(f"{'case':26s} {'variant':32s} {'rule':7s} {'max|dE|':>10s} {'max|dF|':>10s}")
    for label, exe in variants.items():
        for case in cases:
            probe = common.Case(case.name, case.periodic, case.cutoff, case.cell, case.species, case.pos)
            gen.run_driver(exe, [probe], overrides, probe.cutoff)
            for rule in ("mixed", "source"):
                de, df = deviations(case, probe.energy, probe.forces, rule)
                print(f"{case.name:26s} {label:32s} {rule:7s} {de:10.2e} {df:10.2e}")


if __name__ == "__main__":
    main()
