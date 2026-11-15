#!/usr/bin/env python3
"""Check the PLUMED action RGPOT against rgpot itself.

Each case writes one xyz trajectory for ``plumed driver``: the geometry,
then the geometry with single coordinates moved by +-H, then the cell and
atoms strained by +-STRAIN in the six lower-triangular entries. One driver
run gives the energy of every frame and, from DUMPDERIVATIVES, the atomic
and box derivatives of the first. The checks are

* energy and forces against ``rgpot-eval`` on the same geometry,
* forces against central differences of the PLUMED energy,
* box derivatives against central differences over the strains,
* box derivatives against -V sigma from the ``rgpot-eval`` stress,
* PLUMED's own ``--debug-forces``,
* with ``--mpirun``, the same run on 2 and 3 ranks with ``--debug-dd yes``,
  whose energies, derivatives and applied forces must equal the serial
  ones digit for digit,
* and that malformed inputs stop PLUMED with a message naming the fault.

Lengths in the xyz files are nanometres unless a case asks for Angstrom.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# CODATA 2018 e N_A / 1000, the factor rgpot_md_kj_per_ev returns.
KJ_PER_EV = 1.602176634e-19 * 6.02214076e23 / 1000.0
# PLUMED's eV, from UNITS ENERGY=eV, in kJ/mol.
PLUMED_EV = 96.48530749925792
H = 1.0e-4  # Angstrom
STRAIN = 1.0e-5
ROWS = (0, 1, 2, 1, 2, 2)
COLS = (0, 1, 2, 0, 0, 1)
SYMBOLS = {13: "Al", 28: "Ni", 29: "Cu", 18: "Ar", 78: "Pt"}


@dataclass
class Case:
    name: str
    z: list[int]
    xyz: list[list[float]]  # Angstrom
    cell: list[list[float]]  # Angstrom, rows; the reference box for NOPBC
    spec: list[str]  # rgpot-eval words
    action: str  # RGPOT keywords after the label
    nopbc: bool = False
    angstrom_ev: bool = False
    stress: bool = True
    fd_atoms: list[int] = field(default_factory=lambda: list(range(4)))
    extra: list[str] = field(default_factory=list)  # more PLUMED lines


def fcc(a: float, reps: int, rng: random.Random, jitter: float) -> list[list[float]]:
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    out = []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                for b in basis:
                    out.append([(i + b[0]) * a + rng.uniform(-jitter, jitter),
                                (j + b[1]) * a + rng.uniform(-jitter, jitter),
                                (k + b[2]) * a + rng.uniform(-jitter, jitter)])
    return out


def shear(xyz: list[list[float]], length: float, tilt: tuple[float, float, float]):
    """Map a cube of side length to the cell a=(L,0,0), b=(t0,L,0), c=(t1,t2,L)."""
    cell = [[length, 0.0, 0.0], [tilt[0], length, 0.0], [tilt[1], tilt[2], length]]
    out = []
    for x in xyz:
        f = [c / length for c in x]
        out.append([sum(f[r] * cell[r][d] for r in range(3)) for d in range(3)])
    return out, cell


def cluster13(r: float, rng: random.Random) -> list[list[float]]:
    """Icosahedral 13-atom cluster of nearest-neighbour distance r, centred at 0."""
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    scale = r / (1.0 + phi * phi) ** 0.5
    out = [[rng.uniform(-0.02, 0.02) for _ in range(3)]]
    for v in verts:
        out.append([c * scale + rng.uniform(-0.05, 0.05) for c in v])
    return out


def strained(xyz, cell, k: int, e: float):
    i, j = ROWS[k], COLS[k]
    sx = [list(x) for x in xyz]
    for x, orig in zip(sx, xyz):
        x[j] += orig[i] * e
    sc = [list(r) for r in cell]
    for r, orig in zip(sc, cell):
        r[j] += orig[i] * e
    return sx, sc


def det3(m) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def frames(case: Case):
    """The frame list and, for each FD pair, what it differentiates."""
    out = [(case.xyz, case.cell)]
    for a in case.fd_atoms:
        for d in range(3):
            for s in (1.0, -1.0):
                moved = [list(x) for x in case.xyz]
                moved[a][d] += s * H
                out.append((moved, case.cell))
    for k in range(6):
        for s in (1.0, -1.0):
            out.append(strained(case.xyz, case.cell, k, s * STRAIN))
    return out


def write_xyz(path: Path, case: Case, frame_list) -> None:
    scale = 1.0 if case.angstrom_ev else 0.1
    lines = []
    for xyz, cell in frame_list:
        lines.append(str(len(xyz)))
        lines.append(" ".join(f"{c * scale:.17g}" for row in cell for c in row))
        for z, x in zip(case.z, xyz):
            lines.append(f"{SYMBOLS[z]} " + " ".join(f"{c * scale:.17g}" for c in x))
    path.write_text("\n".join(lines) + "\n")


def rgpot_eval(tool: str, case: Case) -> dict:
    lines = [str(len(case.xyz)), " ".join(f"{c:.17g}" for row in case.cell for c in row)]
    for z, x in zip(case.z, case.xyz):
        lines.append(f"{z} " + " ".join(f"{c:.17g}" for c in x))
    out = subprocess.run([tool] + case.spec, input="\n".join(lines) + "\n", text=True,
                         capture_output=True, check=True).stdout
    got = {"forces": [], "stress": None}
    for line in out.splitlines():
        w = line.split()
        if w[0] == "energy":
            got["energy"] = float(w[1])
        elif w[0] == "force":
            got["forces"].append([float(v) for v in w[2:5]])
        elif w[0] == "stress":
            got["stress"] = [float(v) for v in w[1:10]]
    return got


def plumed_input(case: Case, plugin: str, stem: str = "") -> str:
    lines = [f"LOAD FILE={Path(plugin).resolve()}"]
    if case.angstrom_ev:
        lines.append("UNITS LENGTH=A ENERGY=eV")
    lines += [
        f"ene: RGPOT {case.action}",
        *case.extra,
        "BIASVALUE ARG=ene",
        f"PRINT ARG=ene FILE={stem}COLVAR FMT=%.17g",
        f"DUMPDERIVATIVES ARG=ene FILE={stem}DERIV FMT=%.17g",
        "",
    ]
    return "\n".join(lines)


def driver(plumed: list[str], case: Case, work: Path, xyz: str, dat: str, extra=()) -> None:
    cmd = plumed + ["driver", "--plumed", dat, "--ixyz", xyz]
    if case.angstrom_ev:
        cmd += ["--length-units", "A"]
    res = subprocess.run(cmd + list(extra), cwd=work, capture_output=True, text=True,
                         check=False)
    if res.returncode != 0:
        sys.stderr.write(res.stdout[-4000:] + res.stderr[-4000:])
        raise SystemExit(f"{case.name}: plumed driver failed")


def read_values(path: Path) -> list[float]:
    return [float(line.split()[-1]) for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]


def max_abs(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def run_case(case: Case, args, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    plumed = [args.plumed]
    flist = frames(case)
    write_xyz(work / "traj.xyz", case, flist)
    write_xyz(work / "one.xyz", case, flist[:1])
    (work / "plumed.dat").write_text(plumed_input(case, args.plugin))
    driver(plumed, case, work, "traj.xyz", "plumed.dat")
    energies = read_values(work / "COLVAR")
    deriv = read_values(work / "DERIV")
    n = len(case.xyz)
    if len(energies) != len(flist) or len(deriv) < 3 * n + 9:
        raise SystemExit(f"{case.name}: PLUMED wrote {len(energies)} of {len(flist)} frames")
    deriv = deriv[: 3 * n + 9]

    # PLUMED energy per eV and Angstrom per PLUMED length.
    e_unit = KJ_PER_EV / PLUMED_EV if case.angstrom_ev else KJ_PER_EV
    l_unit = 1.0 if case.angstrom_ev else 10.0
    h_plumed = H / l_unit
    ref = rgpot_eval(args.rgpot_eval, case)
    report = {"case": case.name}

    energy = energies[0] / e_unit
    report["dE_rel"] = abs(energy - ref["energy"]) / abs(ref["energy"])
    forces = [[-deriv[3 * a + d] / (e_unit * l_unit) for d in range(3)] for a in range(n)]
    fmax = max(abs(f) for row in ref["forces"] for f in row)
    report["dF_rel"] = max_abs([f for row in forces for f in row],
                               [f for row in ref["forces"] for f in row]) / fmax

    fd_err = 0.0
    at = 1
    for a in case.fd_atoms:
        for d in range(3):
            fd = (energies[at] - energies[at + 1]) / (2.0 * h_plumed)
            fd_err = max(fd_err, abs(fd - deriv[3 * a + d]))
            at += 2
    report["dF_fd_rel"] = fd_err / (fmax * e_unit * l_unit)

    box = deriv[3 * n:]
    w_fd = [0.0] * 9
    for k in range(6):
        w = -(energies[at] - energies[at + 1]) / (2.0 * STRAIN)
        w_fd[3 * ROWS[k] + COLS[k]] = w
        w_fd[3 * COLS[k] + ROWS[k]] = w
        at += 2
    wmax = max(abs(w) for w in w_fd)
    report["dW_fd_rel"] = max_abs(box, w_fd) / wmax
    if ref["stress"] is not None:
        volume = abs(det3(case.cell))
        w_ref = [-volume * s * e_unit for s in ref["stress"]]
        report["dW_sigma_rel"] = max_abs(box, w_ref) / wmax
    elif case.stress:
        raise SystemExit(f"{case.name}: rgpot-eval printed no stress")
    if case.angstrom_ev:
        # PLUMED's eV is smaller than rgpot's, so the value reads high.
        report["plumed_eV_excess"] = energies[0] / ref["energy"] - 1.0

    if not case.nopbc and case.stress:
        # VIRIAL=FD must agree with the stress.
        fd_case = Case(**{**case.__dict__, "action": case.action + " VIRIAL=FD"})
        (work / "plumed_fd.dat").write_text(plumed_input(fd_case, args.plugin, "fd_"))
        driver(plumed, fd_case, work, "one.xyz", "plumed_fd.dat")
        report["dW_FD_vs_stress_rel"] = max_abs(read_values(work / "fd_DERIV")[3 * n:],
                                                box) / wmax

    driver(plumed, case, work, "one.xyz", "plumed.dat", ["--debug-forces", "debug_forces"])
    rows = [line.split() for line in (work / "debug_forces").read_text().splitlines()
            if line.strip() and not line.startswith("#")]
    scale = max(abs(float(r[1])) for r in rows)
    report["debug_forces_rel"] = max(abs(float(r[1]) - float(r[2])) for r in rows) / scale

    if args.mpirun:
        dump = ["--dump-forces", "forces", "--dump-forces-fmt", "%.17g", "--dump-full-virial"]
        (work / "serial.dat").write_text(plumed_input(case, args.plugin, "serial_"))
        driver(plumed, case, work, "traj.xyz", "serial.dat", dump)
        serial = [(work / f"serial_{s}").read_text() for s in ("COLVAR", "DERIV")]
        serial.append((work / "forces").read_text())
        for ranks in (2, 3):
            stem = f"dd{ranks}_"
            (work / f"{stem}.dat").write_text(plumed_input(case, args.plugin, stem))
            mpi = [args.mpirun, "-np", str(ranks), "--oversubscribe", args.plumed]
            driver(mpi, case, work, "traj.xyz", f"{stem}.dat",
                   ["--debug-dd", "yes", "--dump-forces", f"{stem}forces",
                    "--dump-forces-fmt", "%.17g", "--dump-full-virial"])
            got = [(work / f"{stem}{s}").read_text() for s in ("COLVAR", "DERIV")]
            # Every rank dumps the gathered forces to its own file.
            for rank in range(ranks):
                if got + [(work / f"{stem}forces.{rank}").read_text()] != serial:
                    raise SystemExit(f"{case.name}: {ranks} ranks differ from the serial run")
            report[f"dd{ranks}"] = "identical"
    return report


TOL = {
    "dE_rel": 1e-12,
    "dF_rel": 1e-11,
    "dF_fd_rel": 1e-6,
    "dW_fd_rel": 1e-5,
    "dW_sigma_rel": 1e-11,
    "dW_FD_vs_stress_rel": 1e-5,
    "debug_forces_rel": 1e-4,
}


def cases(args) -> list[Case]:
    rng = random.Random(20261115)
    out = []
    ar, ar_cell = shear(fcc(5.26, 3, rng, 0.15), 3 * 5.26, (1.1, -0.7, 0.5))
    out.append(Case(
        "LJ triclinic", [18] * len(ar), ar, ar_cell,
        ["LJ", "u0", "0.0104", "psi", "3.40", "cutoff", "7.5"],
        f"POTENTIAL=LJ OPTIONS=u0=0.0104,psi=3.40,cutoff=7.5 ATOMS=1-{len(ar)} ELEMENTS=Ar"))
    out.append(Case(
        "LJ Angstrom eV", [18] * len(ar), ar, ar_cell,
        ["LJ", "u0", "0.0104", "psi", "3.40", "cutoff", "7.5"],
        f"POTENTIAL=LJ OPTIONS={{u0=0.0104 psi=3.40}} CUTOFF=7.5 ATOMS=1-{len(ar)} ELEMENTS=Ar",
        angstrom_ev=True))
    pt = fcc(3.92, 3, rng, 0.1)
    out.append(Case(
        "Morse", [78] * len(pt), pt, [[3 * 3.92, 0, 0], [0, 3 * 3.92, 0], [0, 0, 3 * 3.92]],
        ["Morse", "cutoff", "6.0"],
        f"POTENTIAL=Morse ATOMS=1-{len(pt)} ELEMENTS=Pt CUTOFF=0.6"))
    if args.eam_alloy:
        cu = fcc(3.6, 4, rng, 0.08)
        cuni, cuni_cell = shear(cu, 4 * 3.6, (0.6, 0.3, -0.4))
        z = [29 if i % 3 else 28 for i in range(len(cuni))]
        order = [i for i in range(len(z)) if z[i] == 29] + [i for i in range(len(z)) if z[i] == 28]
        xyz = [cuni[i] for i in order]
        zs = [z[i] for i in order]
        ncu = zs.count(29)
        out.append(Case(
            "EAMTable CuNi", zs, xyz, cuni_cell,
            ["EAMTable", "file", args.eam_alloy],
            f"POTENTIAL=EAMTable OPTIONS=file={args.eam_alloy} "
            f"SPECIES1=1-{ncu} SPECIES2={ncu + 1}-{len(zs)} ELEMENTS=Cu,Ni"))
    if args.eamal:
        al = fcc(4.05, 3, rng, 0.08)
        out.append(Case(
            "EAMAl, no stress", [13] * len(al), al,
            [[3 * 4.05, 0, 0], [0, 3 * 4.05, 0], [0, 0, 3 * 4.05]],
            ["EAMAl"], f"POTENTIAL=EAMAl ATOMS=1-{len(al)} ELEMENTS=Al", stress=False))
    c13 = cluster13(2.77, rng)
    side = 40.0
    c13 = [[c + side / 2 for c in x] for x in c13]
    out.append(Case(
        "SuttonChen Pt13 NOPBC", [78] * 13, c13, [[side, 0, 0], [0, side, 0], [0, 0, side]],
        ["SuttonChen", "preset", "platinum", "cutoff", "9.0"],
        "POTENTIAL=SuttonChen OPTIONS=preset=platinum,cutoff=9.0 ATOMS=1-13 ELEMENTS=Pt "
        "NOPBC VACUUM=1.0", nopbc=True, fd_atoms=list(range(13))))
    return out


REFUSALS = [
    ("POTENTIAL=NoSuch ATOMS=1-2 ELEMENTS=Ar", "unknown rgpot potential NoSuch"),
    ("POTENTIAL=LJ ATOMS=1-2", "POTENTIAL needs ELEMENTS"),
    ("POTENTIAL=LJ OPTIONS=u0 ATOMS=1-2 ELEMENTS=Ar", "OPTIONS takes key=value pairs"),
    ("POTENTIAL=LJ OPTIONS=depth=1 ATOMS=1-2 ELEMENTS=Ar", "LJ takes no option depth"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Ar,Ar,Ar", "ELEMENTS needs one symbol"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Xx", "Xx is not an element symbol"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Ar NOPBC", "NOPBC needs VACUUM"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Ar U0=1", "U0 and PSI belong to the RGPOT form"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Ar VIRIAL=EXACT", "VIRIAL is AUTO or FD"),
    ("POTENTIAL=LJ ATOMS=1-2 SPECIES1=1 ELEMENTS=Ar", "not both"),
    ("POTENTIAL=LJ SPECIES1=1-2 SPECIES2=2 ELEMENTS=Ar,Ar", "an atom appears twice"),
    ("POTENTIAL=LJ ATOMS=1-2 ELEMENTS=Ar", "needs a periodic box"),
]


def refusals(args, work: Path) -> None:
    """Each line must stop PLUMED with its message. The xyz has no cell."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "dimer.xyz").write_text("2\n0 0 0\nAr 0 0 0\nAr 0.35 0 0\n")
    for keywords, message in REFUSALS:
        (work / "plumed.dat").write_text(
            f"LOAD FILE={Path(args.plugin).resolve()}\nene: RGPOT {keywords}\n"
            "PRINT ARG=ene FILE=COLVAR\n")
        res = subprocess.run([args.plumed, "driver", "--plumed", "plumed.dat", "--ixyz",
                              "dimer.xyz"], cwd=work, capture_output=True, text=True,
                             check=False)
        if res.returncode == 0 or message not in res.stdout + res.stderr:
            raise SystemExit(f"RGPOT {keywords} did not stop with '{message}'")
    print(f"refusals: {len(REFUSALS)} inputs stop with their message")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("plumed")
    p.add_argument("plugin")
    p.add_argument("rgpot_eval")
    p.add_argument("--eam-alloy", default="")
    p.add_argument("--eamal", action="store_true", help="this rgpot has EAMAl")
    p.add_argument("--mpirun", default="")
    p.add_argument("--work", default="plumed-rgpot-action")
    args = p.parse_args()
    if args.mpirun:
        has_mpi = subprocess.run([args.plumed, "config", "has", "mpi"], capture_output=True,
                                 check=False)
        if has_mpi.returncode != 0:
            print("this PLUMED has no MPI; the domain decomposition runs are skipped")
            args.mpirun = ""
    refusals(args, Path(args.work) / "refusals")
    failed = []
    for case in cases(args):
        rep = run_case(case, args, Path(args.work) / case.name.replace(" ", "_").replace(",", ""))
        print(f"{case.name}:")
        for key, value in rep.items():
            if key == "case":
                continue
            text = f"{value:.3e}" if isinstance(value, float) else value
            tol = TOL.get(key)
            bad = tol is not None and not value <= tol
            print(f"  {key:22s} {text}" + (f"  > {tol:g}" if bad else ""))
            if bad:
                failed.append(f"{case.name} {key}")
        excess = rep.get("plumed_eV_excess")
        if excess is not None and abs(excess - (KJ_PER_EV / PLUMED_EV - 1.0)) > 1e-12:
            failed.append(f"{case.name} plumed_eV_excess")
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
