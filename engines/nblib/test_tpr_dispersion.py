#!/usr/bin/env python3
# MIT License
# Copyright 2026 Rohit Goswami
"""Compare the TPR session with ``gmx_d mdrun -rerun`` on two TIP4P waters.

usage: test_tpr_dispersion.py ENGINE.so GMX_D FIXTURE_DIR

The fixture directory holds tip4p.mdp, conf.gro, topol.top and grompp.sh.
grompp.sh is the command, with ``gmx_d`` as its first word. This script
runs that command with GMX_D in a temporary copy of the inputs, then

    gmx_d mdrun -s tip4p.tpr -rerun conf.gro -deffnm rerun -ntmpi 1 -ntomp 1 -noconfout

``mdrun -rerun`` does not rebuild virtual sites unless ``-rerunvsite`` is
set. This run does not set it. The M coordinates in conf.gro are already
the TIP4P virtual site, ``M = O + a*(H1-O) + a*(H2-O)`` with
``a = 0.128012065``, so the frame mdrun reads is the geometry the session
constructs from the oxygen and the hydrogens.

The tpr is not committed. grompp writes it into the temporary directory.
"""

import ctypes as C
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Same SI factors as rgpot/ExternalPot/md_units.hpp.
KJ_PER_EV = 1.602176634e-19 * 6.02214076e23 / 1000.0
TERMS = ("LJ (SR)", "Coulomb (SR)", "Coul. recip.", "Disper. corr.", "Potential")
# Names ``gmx energy`` matches. The menu prints hyphens where the term has a space.
ENERGY_NAMES = ("LJ-(SR)", "Coulomb-(SR)", "Coul.-recip.", "Disper.-corr.", "Potential")


def run(cmd, cwd, env=None):
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc


def grompp(gmx, work, fixture):
    command = (fixture / "grompp.sh").read_text().strip()
    words = command.split()
    if not words or words[0] != "gmx_d":
        raise SystemExit(f"grompp.sh must start with gmx_d, got {command!r}")
    for name in ("tip4p.mdp", "conf.gro", "topol.top"):
        shutil.copy(fixture / name, work / name)
    run([gmx, *words[1:]], work)


def mdrun(gmx, work):
    # No -rerunvsite: virtual sites stay at the coordinates in conf.gro.
    run([gmx, "mdrun", "-s", "tip4p.tpr", "-rerun", "conf.gro", "-deffnm",
         "rerun", "-ntmpi", "1", "-ntomp", "1", "-noconfout"], work)


def energy_term(gmx, work, name):
    proc = subprocess.run(
        [gmx, "energy", "-f", "rerun.edr", "-o", "term.xvg", "-xvg", "none"],
        cwd=work, input=name + "\n\n", text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"gmx energy failed for {name}")
    for line in (work / "term.xvg").read_text().splitlines():
        if line and line[0] not in "#@":
            return float(line.split()[1])
    raise SystemExit(f"gmx energy wrote no value for {name}")


def mdrun_forces(gmx, work):
    env = os.environ.copy()
    env["GMX_PRINT_LONGFORMAT"] = "1"
    proc = subprocess.run([gmx, "dump", "-f", "rerun.trr"], cwd=work, env=env,
                          text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit("gmx dump failed")
    forces = []
    for line in proc.stdout.splitlines():
        if "f[" not in line or "{" not in line:
            continue
        body = line.split("{", 1)[1].split("}", 1)[0]
        forces.append([float(part) for part in body.split(",")])
    if not forces:
        raise SystemExit("rerun.trr has no forces; nstfout must be 1")
    return forces


def session(engine, tpr):
    lib = C.CDLL(engine)
    ops = (C.c_char * 512)()
    ctx, natoms = C.c_void_p(), C.c_size_t()
    err = C.create_string_buffer(4096)
    spec = ("tpr " + str(tpr)).encode()
    create = lib.rgpot_md_session_engine_create
    rc = create(spec, C.c_size_t(len(spec)), b"threads=1", C.byref(ops),
                C.byref(ctx), C.byref(natoms), err, C.c_size_t(len(err)))
    if rc != 0:
        raise SystemExit(f"session create failed: {err.value.decode()}")
    n3 = 3 * natoms.value
    vec = C.c_double * n3
    x, v, f = vec(), vec(), vec()
    potential, kinetic = C.c_double(), C.c_double()
    pointer = C.POINTER(C.c_double)
    tail = [C.c_char_p, C.c_size_t]
    state = C.CFUNCTYPE(C.c_int, C.c_void_p, pointer, pointer, pointer,
                        C.c_size_t, pointer, pointer, *tail)(
        C.cast(C.addressof(ops), C.POINTER(C.c_void_p))[5])
    rc = state(ctx, x, v, f, n3, C.byref(potential), C.byref(kinetic), err, len(err))
    if rc != 0:
        raise SystemExit(f"session state failed: {err.value.decode()}")
    terms = (C.c_double * 5)()
    read = lib.rgpot_nblib_tpr_terms
    read.argtypes = [C.c_void_p, C.POINTER(C.c_double), C.c_size_t]
    read.restype = C.c_int
    if read(ctx, terms, 5) != 0:
        raise SystemExit("the session did not return energy terms")
    # Forces at the C boundary are eV/Angstrom. kJ/mol/nm is that times
    # kJ/eV times 10 Angstrom/nm.
    forces = [f[i] * KJ_PER_EV * 10.0 for i in range(n3)]
    destroy = C.CFUNCTYPE(None, C.c_void_p)(
        C.cast(C.addressof(ops), C.POINTER(C.c_void_p))[7])
    destroy(ctx)
    return list(terms), forces, potential.value * KJ_PER_EV


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_tpr_dispersion.py ENGINE.so GMX_D FIXTURE_DIR")
    engine, gmx, fixture = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    print("M sites in conf.gro are the TIP4P virtual-site construction.")
    print("mdrun -rerun is not asked to rebuild them (-rerunvsite is absent),")
    print("so it uses those stored M coordinates. The session rebuilds M")
    print("from O, H and H on every evaluation; the stored M matches that")
    print("construction.")
    with tempfile.TemporaryDirectory(prefix="tip4p-dispcorr-") as tmp:
        work = Path(tmp)
        grompp(gmx, work, fixture)
        mdrun(gmx, work)
        reference = [energy_term(gmx, work, name) for name in ENERGY_NAMES]
        forces = mdrun_forces(gmx, work)
        got, session_forces, potential = session(engine, work / "tip4p.tpr")
    print(f"{'term':<16} {'session':>16} {'mdrun':>16} {'abs diff':>14}")
    worst_energy = 0.0
    for name, a, b in zip(TERMS, got, reference):
        diff = abs(a - b)
        worst_energy = max(worst_energy, diff)
        print(f"{name:<16} {a:16.8f} {b:16.8f} {diff:14.3e}")
    print(f"{'potential (C API)':<16} {potential:16.8f}")
    # Massive atoms are O, H, H. M is the fourth atom of each molecule.
    if len(forces) % 4 != 0:
        raise SystemExit(f"expected a multiple of 4 atoms, got {len(forces)}")
    index = [i for i in range(len(forces)) if i % 4 != 3]
    if len(session_forces) != 3 * len(index):
        raise SystemExit(
            f"session has {len(session_forces) // 3} atoms, mdrun has {len(index)} massive")
    worst = 0.0
    worst_rel = 0.0
    for j, i in enumerate(index):
        for d in range(3):
            a = session_forces[3 * j + d]
            b = forces[i][d]
            diff = abs(a - b)
            scale = max(abs(a), abs(b))
            rel = diff / scale if scale > 0.0 else 0.0
            worst = max(worst, diff)
            worst_rel = max(worst_rel, rel)
    print(f"largest absolute force difference {worst:.3e} kJ/mol/nm")
    print(f"largest relative force difference {worst_rel:.3e}")
    if abs(reference[3]) < 1e-4:
        raise SystemExit("mdrun dispersion correction is missing")
    if worst_energy > 1e-5:
        raise SystemExit(f"energy differs by {worst_energy} kJ/mol")
    if abs(potential - reference[4]) > 1e-5:
        raise SystemExit("C API potential differs from mdrun")
    if worst > 1e-5:
        raise SystemExit(f"force differs by {worst} kJ/mol/nm")
    print("ok")


if __name__ == "__main__":
    main()
