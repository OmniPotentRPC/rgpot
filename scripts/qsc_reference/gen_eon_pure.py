#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Generate pure-element Quantum Sutton-Chen reference data from real eOn.

The reference is the QSC class from the source snapshot pinned below (QSC was
removed from eOn main). The unmodified
``client/potentials/QSC/QSC.cpp`` is compiled and linked against the eOn
client library built from the same snapshot; ``eon_qsc_driver.cpp`` only feeds
it geometries and prints energies and forces.

Only pure-element cases are produced. For mixtures the pinned eOn QSC reuses pair
parameters across species (QSC.cpp lines 137-157 and 203-223), so it is not a
reference there; see docs/orgmode/reference/qsc.org.

The exact commands used for the committed data file are recorded in its
header, with the machine-specific directories replaced by $EON_SRC, $EON_BUILD,
$PIXI_ENV and $WORKDIR. To regenerate (from the repository root, with an eOn
checkout at the pinned revision built in ``builddir``)::

    export EON_SRC=/path/to/eon-checkout
    git clone https://github.com/TheochemUI/eOn.git "$EON_SRC"
    git -C "$EON_SRC" checkout "$EON_COMMIT"
    (cd "$EON_SRC" && pixi install \\
      && pixi run meson setup builddir \\
           -Dwith_fortran=false -Dwith_cuh2=false -Dwith_tests=false \\
           --buildtype=release \\
      && pixi run meson compile -C builddir)
    RGPOT=$PWD
    (cd "$EON_SRC" && pixi run python \\
       "$RGPOT/scripts/qsc_reference/gen_eon_pure.py" \\
       --eon-src "$EON_SRC" --eon-build "$EON_SRC/builddir" \\
       --out "$RGPOT/CppCore/tests/data/qsc/eon_pure.dat")
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qsc_common as common  # noqa: E402
from qsc_oracle import QSC_TABLE, SYMBOLS  # noqa: E402

EON_COMMIT = "3b2586f543b82cbf8dcdcbd07f2e1126705a3826"
EON_URL = "https://github.com/TheochemUI/eOn"
FREE_BOX = 1000.0
CLUSTER_CUTOFF = 8.0
CELL_CUTOFF = 5.2
CELL_ELEMENTS = (28, 29, 47, 79)


def run(cmd, **kw):
    done = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if done.returncode != 0:
        sys.exit(f"command failed ({done.returncode}): {' '.join(cmd)}\n{done.stderr}")
    return done


def eon_default_table(src):
    """Parse client/potentials/QSC/Parameters.h into {Z: (n, m, eps, c, a)}."""
    text = open(os.path.join(src, "client/potentials/QSC/Parameters.h"), encoding="utf-8").read()
    table = {}
    for m in re.finditer(
        r"\{\s*(\d+),\s*([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+),\s*([\d.e+-]+)\s*\}",
        text,
    ):
        z = int(m.group(1))
        table[z] = tuple(float(m.group(i)) for i in range(2, 7))
    return table


DRIVER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eon_qsc_driver.cpp")


def toolchain_prefix(cxx):
    which = run(["which", cxx]).stdout.strip()
    return os.path.dirname(os.path.dirname(which))


def compile_driver(cxx, opt, src, build, prefix, workdir, name, qsc_cpp=None):
    """Compile eon_qsc_driver.cpp with eOn's QSC.cpp (or a patched copy)."""
    exe = os.path.join(workdir, name)
    cmd = [
        cxx, "-std=c++17", opt, "-I" + src, "-I" + src + "/client",
        "-I" + src + "/client/thirdparty", "-I" + prefix + "/include",
        "-I" + prefix + "/include/eigen3",
        "-include", src + "/client/potentials/QSC/QSC.h",
        "-o", exe, DRIVER_SRC, qsc_cpp or src + "/client/potentials/QSC/QSC.cpp",
        "-L" + build + "/client", "-leonclib", "-leoncbase",
        "-L" + prefix + "/lib", "-lspdlog", "-lfmt",
        "-Wl,-rpath," + build + "/client", "-Wl,-rpath," + prefix + "/lib",
    ]
    run(cmd)
    return exe, " ".join(cmd)


def neutral(line, src, build, prefix, workdir):
    """Replace machine-specific directories in a recorded command line."""
    for path, name in (
        (DRIVER_SRC, "scripts/qsc_reference/eon_qsc_driver.cpp"),
        (workdir, "$WORKDIR"),
        (build, "$EON_BUILD"),
        (src, "$EON_SRC"),
        (prefix, "$PIXI_ENV"),
    ):
        line = line.replace(path, name)
    return line


def build_cases():
    cases = []
    for z, (_, _, _, _, a) in sorted(QSC_TABLE.items()):
        sym = SYMBOLS[z]
        r = 0.9 * a
        cases.append(
            common.Case(
                f"{sym}_dimer", False, CLUSTER_CUTOFF, common.cube(FREE_BOX),
                [z, z], [[0.0, 0.0, 0.0], [r, 0.0, 0.0]],
            )
        )
        cluster = common.jitter(common.cuboctahedron(a, extra=((1.0, 0.0, 0.0), (1.0, 0.5, 0.5))), 0.08, 1000 + z)
        cases.append(
            common.Case(
                f"{sym}_cluster15", False, CLUSTER_CUTOFF, common.cube(FREE_BOX),
                [z] * len(cluster), cluster,
            )
        )
    for z in CELL_ELEMENTS:
        a = QSC_TABLE[z][4]
        pos = common.jitter(common.fcc_supercell(3, a), 0.10, 2000 + z)
        cases.append(
            common.Case(
                f"{SYMBOLS[z]}_cell108", True, CELL_CUTOFF, common.cube(3 * a),
                [z] * len(pos), pos,
            )
        )
    return cases


def driver_input(cases, overrides, cutoff_skin=0.5):
    # eOn takes one global cutoff; cases are grouped by cutoff.
    lines = [str(len(overrides))]
    for z, row in sorted(overrides.items()):
        lines.append(f"{z} " + " ".join(repr(v) for v in row))
    return lines


def run_driver(binary, cases, overrides, cutoff):
    sel = [c for c in cases if c.cutoff == cutoff]
    lines = driver_input(sel, overrides)
    lines.append(f"{cutoff!r} 0.5")
    lines.append(str(len(sel)))
    for c in sel:
        lines.append(f"{c.name} {len(c.species)} {c.cell[0][0]!r} {c.cell[1][1]!r} {c.cell[2][2]!r}")
        for z, p in zip(c.species, c.pos):
            lines.append(f"{z} " + " ".join(repr(v) for v in p))
    out = run([binary], input="\n".join(lines) + "\n").stdout.splitlines()
    it = iter(out)
    for c in sel:
        header = next(it).split()
        assert header[0] == "case" and header[1] == c.name
        c.energy = float(next(it).split()[1])
        c.forces = [[float(x) for x in next(it).split()[1:4]] for _ in c.species]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eon-src", required=True, help="pinned eOn checkout")
    ap.add_argument("--eon-build", required=True, help="meson build dir of that checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--opt", default="-O2")
    args = ap.parse_args(argv)

    src = os.path.abspath(args.eon_src)
    build = os.path.abspath(args.eon_build)
    head = run(["git", "-C", src, "rev-parse", "HEAD"]).stdout.strip()
    if head != EON_COMMIT:
        sys.exit(f"eOn checkout is at {head}, need {EON_COMMIT}")

    cxx_version = run([args.cxx, "--version"]).stdout.splitlines()[0]
    prefix = toolchain_prefix(args.cxx)
    workdir = tempfile.mkdtemp(prefix="qsc-eon-")
    exe, compile_line = compile_driver(args.cxx, args.opt, src, build, prefix, workdir, "eon_qsc_driver")
    compile_line = neutral(compile_line, src, build, prefix, workdir)

    eon_table = eon_default_table(src)
    overrides = {}
    notes = []
    for z, row in sorted(QSC_TABLE.items()):
        if eon_table.get(z) != row:
            overrides[z] = row
            notes.append(
                f"{SYMBOLS[z]} (Z={z}): eOn default {eon_table.get(z)} replaced by {row} via set_qsc_parameter"
            )

    cases = build_cases()
    for cutoff in sorted({c.cutoff for c in cases}):
        run_driver(exe, cases, overrides, cutoff)

    # QSC.cpp leaves some qsc_parameters locals uninitialised and relies on
    # stack reuse, so check that an unoptimised build gives identical numbers.
    exe0, _ = compile_driver(args.cxx, "-O0", src, build, prefix, workdir, "eon_qsc_driver_O0")
    check = build_cases()
    for cutoff in sorted({c.cutoff for c in check}):
        run_driver(exe0, check, overrides, cutoff)
    same = all(
        a.energy == b.energy and a.forces == b.forces for a, b in zip(cases, check)
    )
    if not same:
        sys.exit("eOn QSC gives different pure-element results at -O0 and " + args.opt)

    digest = hashlib.sha256(open(DRIVER_SRC, "rb").read()).hexdigest()
    header = [
        "MIT License",
        "Copyright 2023--present rgpot developers",
        "",
        "Pure-element Quantum Sutton-Chen reference energies (eV) and forces (eV/Angstrom)",
        f"computed by eOn QSC, {EON_URL}",
        "pinned source snapshot recorded by scripts/qsc_reference/gen_eon_pure.py.",
        "Mixtures are deliberately absent: QSC.cpp lines 137-157 and",
        "203-223 reuse pair parameters across species, so eOn is not a reference there.",
        "",
        "Generated by scripts/qsc_reference/gen_eon_pure.py with:",
        "  $EON_SRC is the eOn checkout, $EON_BUILD its meson build directory, $PIXI_ENV the",
        "  default pixi environment of that checkout (toolchain, spdlog, fmt, eigen) and",
        "  $WORKDIR a scratch directory.",
        f"  compiler: {cxx_version}",
        f"  command : {compile_line}",
        f"  driver  : scripts/qsc_reference/eon_qsc_driver.cpp sha256 {digest}",
        f"  eOn libs: meson setup builddir -Dwith_fortran=false -Dwith_cuh2=false -Dwith_tests=false --buildtype=release",
        f"  run     : cutoff {CLUSTER_CUTOFF} (clusters) and {CELL_CUTOFF} (cells), verlet skin 0.5, one fresh QSC per case",
        f"  check   : -O0 build of the same driver gives bit-identical numbers to {args.opt}",
        "Parameters below are the table the case was evaluated with. eOn's own",
        "client/potentials/QSC/Parameters.h differs from it as follows:",
    ]
    header += ["  " + n for n in notes] or ["  (no differences)"]
    header += [
        "Cluster cases are non-periodic in rgpot (box 1000 in eOn, minimum image is inert).",
        "Cell cases are periodic orthorhombic with cutoff < half the box edge, so eOn's",
        "single minimum image and rgpot's neighbour search enumerate the same pairs.",
    ]
    common.write_cases(args.out, header, QSC_TABLE, cases)
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
