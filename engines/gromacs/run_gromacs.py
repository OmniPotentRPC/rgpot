#!/usr/bin/env python3
"""Run GROMACS steps whose only energy is the RGPOT bias.

GROMACS 2026 in this recipe has no NNPot module. It does accept
``gmx mdrun -plumed``, and the PLUMED action calls rgpot. Every topology
has zero Lennard-Jones and zero charge, so the bias is the whole
potential and the forces in the trajectory are the rgpot forces.

The cases are the shifted Lennard-Jones dimer of the original RGPOT
keywords, checked against ``rgpot-lj``, and with ``--rgpot-eval`` a Morse
Pt13 cluster under NOPBC and an EAMTable Cu7Ni6 cluster in GROMACS's
periodic box, whose bias and forces are checked against ``rgpot-eval``.
mdrun hands PLUMED single precision coordinates. On those rounded
coordinates rgpot-eval gives the bias to rounding; on the coordinates of
the gro file it differs by about 1e-7. ``gmx dump`` prints forces with six
significant digits, which bounds their agreement.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
from pathlib import Path

# CODATA 2018 e N_A / 1000, the factor rgpot_md_kj_per_ev returns.
KJ_PER_EV = 1.602176634e-19 * 6.02214076e23 / 1000.0
MASS = {"Ar": 39.948, "Pt": 195.08, "Cu": 63.546, "Ni": 58.693}
Z = {"Ar": 18, "Pt": 78, "Cu": 29, "Ni": 28}

MDP = """
integrator = md
dt = 0.001
nsteps = 0
continuation = yes
cutoff-scheme = Verlet
nstlist = 1
pbc = xyz
coulombtype = Cut-off
rcoulomb = 1.2
rvdw = 1.2
rlist = 1.2
vdw-modifier = None
DispCorr = no
nstfout = 1
"""


def topology(symbols: list[str]) -> str:
    types = sorted(set(symbols))
    lines = ["[ defaults ]", "1 2 yes 0.5 0.8333", "", "[ atomtypes ]"]
    lines += [f"{s} {MASS[s]} 0.0 A 0.10 0.0" for s in types]
    for s in types:
        lines += ["", "[ moleculetype ]", f"{s}X 1", "", "[ atoms ]",
                  f"1 {s} 1 {s}X {s} 1 0.0"]
    lines += ["", "[ system ]", "rgpot bias", "", "[ molecules ]"]
    # One entry per run of equal symbols keeps the gro order.
    run, count = symbols[0], 0
    for s in symbols + [None]:
        if s == run:
            count += 1
            continue
        lines.append(f"{run}X {count}")
        run, count = s, 1
    return "\n".join(lines) + "\n"


def gro(symbols: list[str], nm: list[list[float]], box: float) -> str:
    lines = ["rgpot bias", str(len(symbols))]
    for i, (s, x) in enumerate(zip(symbols, nm), start=1):
        lines.append(f"{i:5d}{s + 'X':<5s}{s:>5s}{i:5d}{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}")
    lines.append(f"{box:10.5f}{box:10.5f}{box:10.5f}")
    return "\n".join(lines) + "\n"


def icosahedron(r: float, centre: float) -> list[list[float]]:
    """13 atoms, nearest-neighbour distance r (nm), slightly distorted."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    scale = r / (1.0 + phi * phi) ** 0.5
    out = [[centre, centre, centre]]
    for k, v in enumerate(verts):
        bump = 1.0 + 0.02 * ((k % 5) - 2)
        out.append([round(centre + c * scale * bump, 3) for c in v])
    return out


def mdrun(work: Path, plumed_dat: str, symbols, nm, box: float, env, args) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "plumed.dat").write_text(plumed_dat)
    (work / "topol.top").write_text(topology(symbols))
    (work / "conf.gro").write_text(gro(symbols, nm, box))
    (work / "md.mdp").write_text(MDP)
    subprocess.run(
        [args.gmx, "grompp", "-f", "md.mdp", "-c", "conf.gro", "-p", "topol.top", "-o", "run.tpr",
         "-maxwarn", "2"],
        cwd=work, check=True, env=env, capture_output=True,
    )
    subprocess.run(
        [args.gmx, "mdrun", "-s", "run.tpr", "-deffnm", "run", "-plumed", "plumed.dat",
         "-nsteps", "0", "-ntmpi", "1", "-ntomp", "1"],
        cwd=work, check=True, env=env, capture_output=True,
    )


def bias(work: Path) -> float:
    values = [line for line in (work / "COLVAR").read_text().splitlines()
              if line and not line.startswith("#")]
    if not values:
        raise SystemExit(f"{work.name}: PLUMED wrote no energy")
    return float(values[-1].split()[-1])


def trr_forces(work: Path, env, gmx: str) -> list[list[float]]:
    out = subprocess.run([gmx, "dump", "-f", "run.trr"], cwd=work, check=True, env=env,
                         capture_output=True, text=True).stdout
    pat = re.compile(r"f\[\s*\d+\]=\{\s*([^,]+),\s*([^,]+),\s*([^}]+)\}")
    return [[float(v) for v in m.groups()] for m in pat.finditer(out)]


def single(x: float) -> float:
    return struct.unpack("f", struct.pack("f", x))[0]


def rgpot_eval(tool: str, spec: list[str], symbols, nm, box: float, rounding=float) -> dict:
    lines = [str(len(symbols)), f"{10 * box} 0 0 0 {10 * box} 0 0 0 {10 * box}"]
    for s, x in zip(symbols, nm):
        lines.append(f"{Z[s]} " + " ".join(f"{10 * rounding(c):.17g}" for c in x))
    out = subprocess.run([tool] + spec, input="\n".join(lines) + "\n", text=True,
                         capture_output=True, check=True).stdout
    got = {"forces": []}
    for line in out.splitlines():
        w = line.split()
        if w[0] == "energy":
            got["energy"] = float(w[1])
        elif w[0] == "force":
            got["forces"].append([float(v) for v in w[2:5]])
    return got


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("rgpot_lj")
    p.add_argument("plugin")
    p.add_argument("work")
    p.add_argument("--rgpot-eval", default="")
    p.add_argument("--eam-alloy", default="")
    p.add_argument("--gmx", default="gmx")
    args = p.parse_args()
    work = Path(args.work)
    load = f"LOAD FILE={Path(args.plugin).resolve()}"
    env = os.environ.copy()
    prefix = env.get("CONDA_PREFIX")
    if prefix and "PLUMED_KERNEL" not in env:
        kernel = Path(prefix) / "lib" / "libplumedKernel.so"
        if kernel.is_file():
            env["PLUMED_KERNEL"] = str(kernel)

    # 1 eV well, 0.8 nm cutoff, 0.1 nm zero crossing.
    factor = float(subprocess.check_output([args.rgpot_lj, "--kJ-per-eV"], text=True).strip())
    dimer = work / "dimer"
    mdrun(dimer, "\n".join([
        load,
        f"ene: RGPOT ATOMS=1,2 U0={factor:.16e} CUTOFF=0.8 PSI=0.1",
        "b: BIASVALUE ARG=ene",
        "PRINT ARG=ene FILE=COLVAR FMT=%.17g",
        "",
    ]), ["Ar", "Ar"], [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], 4.0, env, args)
    ev = bias(dimer) / factor
    direct = subprocess.check_output(
        [args.rgpot_lj], input=(Path(__file__).resolve().parents[1] / "compare_geom.txt").read_text(),
        text=True,
    )
    ref = float(next(line.split()[1] for line in direct.splitlines() if line.startswith("energy")))
    if abs(ev - ref) > 1e-6:
        raise SystemExit(f"gromacs bias {ev} eV, rgpot {ref} eV")
    print(f"gromacs bias {ev:.10f} eV")

    if not args.rgpot_eval:
        return
    box = 5.0
    cases = [(
        "Morse Pt13 NOPBC", ["Pt"] * 13, icosahedron(0.277, 2.5), ["Morse"],
        "POTENTIAL=Morse ATOMS=1-13 ELEMENTS=Pt NOPBC VACUUM=1.0",
    )]
    if args.eam_alloy:
        cases.append((
            "EAMTable Cu7Ni6", ["Cu"] * 7 + ["Ni"] * 6, icosahedron(0.252, 2.5),
            ["EAMTable", "file", args.eam_alloy],
            f"POTENTIAL=EAMTable OPTIONS=file={args.eam_alloy} SPECIES1=1-7 SPECIES2=8-13 "
            "ELEMENTS=Cu,Ni",
        ))
    for name, symbols, nm, spec, keywords in cases:
        case_work = work / name.split()[0].lower()
        mdrun(case_work, "\n".join([
            load, f"ene: RGPOT {keywords}", "b: BIASVALUE ARG=ene",
            "PRINT ARG=ene FILE=COLVAR FMT=%.17g", "",
        ]), symbols, nm, box, env, args)
        ref = rgpot_eval(args.rgpot_eval, spec, symbols, nm, box, single)
        exact = rgpot_eval(args.rgpot_eval, spec, symbols, nm, box)["energy"]
        ev = bias(case_work) / KJ_PER_EV
        de = abs(ev - ref["energy"]) / abs(ref["energy"])
        de_gro = abs(ev - exact) / abs(exact)
        got = trr_forces(case_work, env, args.gmx)
        want = [[f * KJ_PER_EV * 10.0 for f in row] for row in ref["forces"]]
        fmax = max(abs(f) for row in want for f in row)
        df = max(abs(a - b) for ra, rb in zip(got, want) for a, b in zip(ra, rb)) / fmax
        print(f"{name}: gromacs bias {ev:.10f} eV, rgpot {ref['energy']:.10f} eV, "
              f"|dE|/|E| {de:.2e} ({de_gro:.2e} on the gro coordinates), "
              f"max|dF|/max|F| {df:.2e}")
        if len(got) != len(symbols) or de > 1e-12 or de_gro > 1e-6 or df > 1e-5:
            raise SystemExit(f"{name}: GROMACS and rgpot disagree")


if __name__ == "__main__":
    main()
