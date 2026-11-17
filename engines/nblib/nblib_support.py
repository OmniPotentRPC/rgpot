"""Shared fixtures of the nblib tests: topologies, systems, the library.

Everything here is in the units of the topology text (nm, kJ/mol, amu, e)
until a call crosses the C ABI, which takes Angstrom, ps, eV and amu.
"""

from __future__ import annotations

import ctypes as C
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = os.environ.get("RGPOT_NBLIB_ENGINE", "")
PREFIX = "rgpot_nblib"

# CODATA 2018 as in CppCore/rgpot/ExternalPot/md_units.hpp.
KJ_PER_EV = 1.602176634e-19 * 6.02214076e23 / 1000.0
ANGSTROM_PER_NM = 10.0
# 1 eV/(Angstrom amu) in Angstrom/ps^2, and kT of 1 eV in amu Angstrom^2/ps^2.
ACCELERATION = KJ_PER_EV * ANGSTROM_PER_NM * ANGSTROM_PER_NM
BOLTZMANN_EV = 8.617333262e-5

EPS = np.finfo(np.float64).eps


def load_mdsession():
    path = HERE.parents[1] / "python" / "rgpot" / "mdsession.py"
    spec = importlib.util.spec_from_file_location("rgpot_mdsession", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mdsession = load_mdsession()


def engine():
    return mdsession.MDEngine(ENGINE, PREFIX)


class Potential:
    """ctypes face of rgpot_nblib_potential_*."""

    def __init__(self, topology: str):
        self.lib = C.CDLL(ENGINE)
        lib = self.lib
        dp = C.POINTER(C.c_double)
        lib.rgpot_nblib_potential_create.restype = C.c_void_p
        lib.rgpot_nblib_potential_create.argtypes = [C.c_char_p, C.c_char_p, C.c_size_t]
        lib.rgpot_nblib_potential_free.argtypes = [C.c_void_p]
        lib.rgpot_nblib_potential_atom_count.restype = C.c_size_t
        lib.rgpot_nblib_potential_atom_count.argtypes = [C.c_void_p]
        lib.rgpot_nblib_potential_cutoff.restype = C.c_double
        lib.rgpot_nblib_potential_cutoff.argtypes = [C.c_void_p]
        lib.rgpot_nblib_potential_energy_forces.restype = C.c_int
        lib.rgpot_nblib_potential_energy_forces.argtypes = [
            C.c_void_p, dp, C.c_size_t, dp, dp, dp, dp, C.c_char_p, C.c_size_t,
        ]
        lib.rgpot_nblib_last_error.restype = C.c_char_p
        err = C.create_string_buffer(512)
        self.handle = lib.rgpot_nblib_potential_create(topology.encode(), err, 512)
        if not self.handle:
            raise RuntimeError(err.value.decode())
        self.atoms = lib.rgpot_nblib_potential_atom_count(self.handle)

    def close(self):
        if self.handle:
            self.lib.rgpot_nblib_potential_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def evaluate(self, positions_angstrom, box_angstrom):
        """energy (eV), forces (eV/Angstrom, (n, 3)), virial (eV, (3, 3))."""
        x = np.ascontiguousarray(positions_angstrom, dtype=np.float64).reshape(-1)
        box = np.ascontiguousarray(box_angstrom, dtype=np.float64).reshape(9)
        forces = np.empty_like(x)
        energy = C.c_double()
        virial = np.empty(9)
        dp = C.POINTER(C.c_double)
        err = C.create_string_buffer(512)
        code = self.lib.rgpot_nblib_potential_energy_forces(
            self.handle,
            x.ctypes.data_as(dp),
            self.atoms,
            box.ctypes.data_as(dp),
            forces.ctypes.data_as(dp),
            C.byref(energy),
            virial.ctypes.data_as(dp),
            err,
            512,
        )
        if code != 0:
            raise RuntimeError(err.value.decode())
        return energy.value, forces.reshape(-1, 3), virial.reshape(3, 3)


@dataclass
class System:
    name: str
    topology: str
    positions_nm: np.ndarray  # (n, 3)
    box_nm: float
    masses: np.ndarray  # (n,) amu
    cutoff_nm: float
    gmx_top: str
    charged: bool
    gro_residue: str = "X"
    gro_atoms: tuple = ("X",)

    @property
    def positions(self):
        return self.positions_nm * ANGSTROM_PER_NM

    @property
    def box(self):
        edge = self.box_nm * ANGSTROM_PER_NM
        return np.diag([edge, edge, edge])


ARGON_C6 = 4 * 0.9961 * 0.34**6
ARGON_C12 = 4 * 0.9961 * 0.34**12


def argon(cells: int = 3, cutoff_nm: float = 0.7, seed: int = 1) -> System:
    spacing = 0.53
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    sites = [
        (np.array([i, j, k]) + b) * spacing
        for i in range(cells)
        for j in range(cells)
        for k in range(cells)
        for b in basis
    ]
    rng = np.random.default_rng(seed)
    x = np.array(sites) + rng.normal(0.0, 0.01, (len(sites), 3))
    n = len(sites)
    topology = f"""nblib 1
cutoff {cutoff_nm}
coulomb none
atomtype Ar 39.948 {ARGON_C6!r} {ARGON_C12!r}
molecule argon {n}
  atom Ar Ar 0.0
end
"""
    gmx_top = f"""[ defaults ]
1 1 no 1 1
[ atomtypes ]
Ar 18 39.948 0.0 A {ARGON_C6!r} {ARGON_C12!r}
[ moleculetype ]
AR 1
[ atoms ]
1 Ar 1 AR AR 1 0.0
[ system ]
argon
[ molecules ]
AR {n}
"""
    return System(
        "argon", topology, x, cells * spacing, np.full(n, 39.948), cutoff_nm,
        gmx_top, False, "AR", ("AR",),
    )


# SPC/E charges and Lennard-Jones, with the flexible bond and angle of the
# GROMACS flexible water; nblib has no constraints.
OW_C6 = 0.0026171
OW_C12 = 2.6331e-06
Q_O = -0.8476
Q_H = 0.4238
BOND_K = 345000.0
BOND_R0 = 0.1
ANGLE_K = 383.0
ANGLE_THETA0 = 109.47


def water(side: int = 4, cutoff_nm: float = 0.55, seed: int = 2) -> System:
    spacing = 0.31
    rng = np.random.default_rng(seed)
    half = np.radians(ANGLE_THETA0) / 2.0
    template = np.array(
        [
            [0.0, 0.0, 0.0],
            [BOND_R0 * np.sin(half), BOND_R0 * np.cos(half), 0.0],
            [-BOND_R0 * np.sin(half), BOND_R0 * np.cos(half), 0.0],
        ]
    )
    atoms = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = rng.uniform(0.0, 0.5)
                w, (a, b, c) = np.cos(angle / 2), np.sin(angle / 2) * axis
                rotation = np.array(
                    [
                        [1 - 2 * (b * b + c * c), 2 * (a * b - c * w), 2 * (a * c + b * w)],
                        [2 * (a * b + c * w), 1 - 2 * (a * a + c * c), 2 * (b * c - a * w)],
                        [2 * (a * c - b * w), 2 * (b * c + a * w), 1 - 2 * (a * a + b * b)],
                    ]
                )
                centre = (np.array([i, j, k]) + 0.5) * spacing
                atoms.append(centre + template @ rotation.T)
    x = np.concatenate(atoms) + rng.normal(0.0, 0.004, (3 * side**3, 3))
    n = side**3
    topology = f"""nblib 1
cutoff {cutoff_nm}
coulomb cutoff
atomtype OW 15.9994 {OW_C6} {OW_C12}
atomtype HW 1.008 0.0 0.0
molecule spce {n}
  atom O OW {Q_O}
  atom H1 HW {Q_H}
  atom H2 HW {Q_H}
  exclude O H1
  exclude O H2
  exclude H1 H2
  bond harmonic O H1 {BOND_K} {BOND_R0}
  bond harmonic O H2 {BOND_K} {BOND_R0}
  angle harmonic H1 O H2 {ANGLE_K} {ANGLE_THETA0}
end
"""
    gmx_top = f"""[ defaults ]
1 1 no 1 1
[ atomtypes ]
OW 8 15.9994 0.0 A {OW_C6} {OW_C12}
HW 1 1.008 0.0 A 0.0 0.0
[ moleculetype ]
SOL 3
[ atoms ]
1 OW 1 SOL OW 1 {Q_O}
2 HW 1 SOL HW1 1 {Q_H}
3 HW 1 SOL HW2 1 {Q_H}
[ bonds ]
1 2 1 {BOND_R0} {BOND_K}
1 3 1 {BOND_R0} {BOND_K}
[ angles ]
2 1 3 1 {ANGLE_THETA0} {ANGLE_K}
[ system ]
water
[ molecules ]
SOL {n}
"""
    masses = np.tile([15.9994, 1.008, 1.008], n)
    return System(
        "water", topology, x, side * spacing, masses, cutoff_nm, gmx_top, True,
        "SOL", ("OW", "HW1", "HW2"),
    )


CT_C6 = 0.0023
CT_C12 = 2.0e-06


def chain(count: int = 4, cutoff_nm: float = 0.6, seed: int = 3, drop=()) -> System:
    """Seven-atom chains with every bonded form the topology text takes."""
    length = 7
    rng = np.random.default_rng(seed)
    edge = 1.4
    atoms = []
    for c in range(count):
        origin = np.array([0.15 + 0.3 * (c % 2), 0.2 + 0.6 * (c // 2), 0.3 + 0.5 * (c % 2)])
        for i in range(length):
            atoms.append(origin + [0.12 * i + 0.15, 0.06 * (i % 2), 0.03 * ((i // 2) % 2)])
    x = np.array(atoms) + rng.normal(0.0, 0.006, (count * length, 3))
    names = [f"C{i + 1}" for i in range(length)]
    near = [(i, j) for i in range(length) for j in range(i + 1, length) if j - i <= 3]
    lines = [
        "nblib 1", f"cutoff {cutoff_nm}", "coulomb none",
        f"atomtype CT 14.0 {CT_C6} {CT_C12}", f"molecule chain {count}",
    ]
    lines += [f"  atom {n} CT 0.0" for n in names]
    lines += [f"  exclude {names[i]} {names[j]}" for i, j in near]
    bonds = [
        ("harmonic", 0, 1, "250000 0.15"), ("harmonic", 1, 2, "250000 0.15"),
        ("harmonic", 2, 3, "250000 0.15"), ("g96", 3, 4, "7.0e6 0.15"),
        ("morse", 4, 5, "400 20.0 0.15"), ("harmonic", 5, 6, "250000 0.15"),
    ]
    bonds = [t for t in bonds if t[0] not in drop]
    lines += [f"  bond {k} {names[a]} {names[b]} {p}" for k, a, b, p in bonds]
    angles = [
        ("harmonic", 0, 1, 2, "500 111"), ("harmonic", 1, 2, 3, "500 111"),
        ("g96", 2, 3, 4, "450 112"), ("harmonic", 3, 4, 5, "500 111"),
        ("harmonic", 4, 5, 6, "500 109"),
    ]
    angles = [t for t in angles if t[0] + "angle" not in drop]
    lines += [f"  angle {k} {names[a]} {names[b]} {names[c_]} {p}" for k, a, b, c_, p in angles]
    dihedrals = [
        ("proper", 0, 1, 2, 3, "10 5.0 3"), ("rb", 1, 2, 3, 4, "9.28 12.16 -13.12 -3.06 26.24 -31.5"),
        ("proper", 2, 3, 4, 5, "180 2.5 2"), ("improper", 3, 4, 5, 6, "0 40"),
    ]
    dihedrals = [t for t in dihedrals if t[0] not in drop]
    lines += [
        f"  dihedral {k} {names[a]} {names[b]} {names[c_]} {names[d]} {p}"
        for k, a, b, c_, d, p in dihedrals
    ]
    pairs = [(0, 3), (1, 4), (2, 5), (3, 6)]
    lines += [f"  pair14 {names[a]} {names[b]} {CT_C6} {CT_C12}" for a, b in pairs]
    lines.append("end")
    topology = "\n".join(lines) + "\n"
    top = [
        "[ defaults ]", "1 1 no 1 1", "[ atomtypes ]",
        f"CT 6 14.0 0.0 A {CT_C6} {CT_C12}", "[ moleculetype ]", "CHN 3", "[ atoms ]",
    ]
    top += [f"{i + 1} CT 1 CHN {names[i]} {i + 1} 0.0" for i in range(length)]
    top += ["[ bonds ]"]
    gmx_bond = {"harmonic": 1, "g96": 2, "morse": 3}
    for k, a, b, p in bonds:
        parts = p.split()
        if k == "morse":
            depth, beta, r0 = parts
            parts = [r0, depth, beta]
        else:
            parts = [parts[1], parts[0]]
        top.append(f"{a + 1} {b + 1} {gmx_bond[k]} " + " ".join(parts))
    top += ["[ pairs ]"] + [f"{a + 1} {b + 1} 1 {CT_C6} {CT_C12}" for a, b in pairs]
    top += ["[ angles ]"]
    for k, a, b, c_, p in angles:
        theta0, kk = p.split()[1], p.split()[0]
        top.append(f"{a + 1} {b + 1} {c_ + 1} {1 if k == 'harmonic' else 2} {theta0} {kk}")
    top += ["[ dihedrals ]"]
    for k, a, b, c_, d, p in dihedrals:
        func = {"proper": 1, "improper": 2, "rb": 3}[k]
        top.append(f"{a + 1} {b + 1} {c_ + 1} {d + 1} {func} {p}")
    top += ["[ system ]", "chains", "[ molecules ]", f"CHN {count}"]
    gmx_top = "\n".join(top) + "\n"
    return System(
        "chain", topology, x, edge, np.full(count * length, 14.0), cutoff_nm,
        gmx_top, False, "CHN", tuple(names),
    )


def maxwell(masses_amu, temperature_k: float, seed: int) -> np.ndarray:
    """Velocities in Angstrom/ps with kT per coordinate and no net momentum."""
    rng = np.random.default_rng(seed)
    kt = BOLTZMANN_EV * temperature_k * ACCELERATION  # amu Angstrom^2/ps^2
    m = np.repeat(masses_amu, 3).reshape(-1, 3)
    v = rng.normal(size=m.shape) * np.sqrt(kt / m)
    momentum = (m * v).sum(axis=0) / m[:, 0].sum()
    return (v - momentum).reshape(-1)


# ---------------------------------------------------------------------------
# GROMACS reference: grompp, mdrun -rerun and mdrun -nsteps 0 on one frame.
# ---------------------------------------------------------------------------

import re
import struct
import subprocess

MDP = """integrator = md
nsteps = 0
continuation = yes
cutoff-scheme = Verlet
coulombtype = Reaction-field
epsilon-rf = 1
epsilon-r = 1
vdwtype = Cut-off
vdw-modifier = Potential-shift
coulomb-modifier = Potential-shift
rvdw = {cutoff}
rcoulomb = {cutoff}
rlist = {cutoff}
verlet-buffer-tolerance = -1
DispCorr = no
nstcalcenergy = 1
nstenergy = 1
nstfout = 1
nstxout = 1
pbc = xyz
"""


def gro_text(system: System):
    """The frame as .gro text with eight decimals, and the positions as
    GROMACS reads them back."""
    n = len(system.positions_nm)
    edge = system.box_nm
    lines = [system.name, f"{n}"]
    parsed = np.empty((n, 3))
    for i, p in enumerate(system.positions_nm):
        fields = [f"{c:13.8f}" for c in p]
        parsed[i] = [float(f) for f in fields]
        per = len(system.gro_atoms)
        residue = i // per + 1
        name = system.gro_atoms[i % per]
        lines.append(
            f"{residue:5d}{system.gro_residue:<5s}{name:>5s}{i + 1:5d}"
            + "".join(fields)
        )
    lines.append(f"{edge:14.8f}{edge:14.8f}{edge:14.8f}")
    return "\n".join(lines) + "\n", parsed


def read_trr(path):
    """Positions, forces and box of the first frame of a .trr (big-endian
    XDR), in the file's precision."""
    data = Path(path).read_bytes()
    pos = 0

    def ints(count):
        nonlocal pos
        values = struct.unpack(">" + "i" * count, data[pos : pos + 4 * count])
        pos += 4 * count
        return values

    magic, = ints(1)
    assert magic == 1993, "not a trr file"
    ints(1)  # version of the string
    length, = ints(1)
    pos += (length + 3) // 4 * 4
    ir, e, box, vir, pres, top, sym, xs, vs, fs, natoms, step, nre = ints(13)
    real = ">d" if box == 72 or (natoms and xs // (3 * natoms) == 8) else ">f"
    width = struct.calcsize(real)
    pos += 2 * width  # t, lambda

    def reals(count):
        nonlocal pos
        values = np.frombuffer(data, dtype=real, count=count, offset=pos)
        pos += count * width
        return values.astype(np.float64)

    cell = reals(9).reshape(3, 3) if box else None
    reals(vir // width)
    reals(pres // width)
    x = reals(xs // width).reshape(-1, 3) if xs else None
    reals(vs // width)
    f = reals(fs // width).reshape(-1, 3) if fs else None
    return {"x": x, "f": f, "box": cell}


def edr_terms(gmx, edr):
    """name -> value of the first frame, as `gmx dump -e` prints it."""
    text = subprocess.run(
        [gmx, "dump", "-e", str(edr)], capture_output=True, text=True, check=True
    ).stdout
    block = text.split("Sum Energy", 1)[1]
    terms = {}
    for line in block.splitlines():
        match = re.match(r"\s*(\S.*?)\s{2,}(-?\d\.\d+e[+-]\d+)", line)
        if match:
            terms[match.group(1)] = float(match.group(2))
    return terms


def gmx_reference(gmx, system: System, workdir: Path):
    """Energy (kJ/mol), forces (kJ/mol/nm), virial (kJ/mol) and the positions
    as GROMACS read them, from mdrun -rerun and a zero-step mdrun."""
    workdir.mkdir(parents=True, exist_ok=True)
    gro, parsed = gro_text(system)
    (workdir / "conf.gro").write_text(gro)
    (workdir / "sys.top").write_text(system.gmx_top)
    (workdir / "sys.mdp").write_text(MDP.format(cutoff=system.cutoff_nm))

    def run(*args):
        done = subprocess.run(
            [gmx, *args], cwd=workdir, capture_output=True, text=True
        )
        if done.returncode != 0:
            raise RuntimeError(f"gmx {' '.join(args)} failed:\n{done.stderr[-2000:]}")

    run("grompp", "-f", "sys.mdp", "-c", "conf.gro", "-p", "sys.top", "-o", "sys.tpr",
        "-po", "out.mdp", "-maxwarn", "1")
    run("mdrun", "-s", "sys.tpr", "-rerun", "conf.gro", "-nt", "1",
        "-e", "rerun.edr", "-o", "rerun.trr", "-g", "rerun.log")
    frame = read_trr(workdir / "rerun.trr")
    energy = edr_terms(gmx, workdir / "rerun.edr")["Potential"]
    run("mdrun", "-s", "sys.tpr", "-nt", "1", "-e", "md.edr", "-o", "md.trr",
        "-g", "md.log", "-c", "md.gro")
    md = edr_terms(gmx, workdir / "md.edr")
    virial = np.array(
        [[md[f"Vir-{a}{b}"] for b in "XYZ"] for a in "XYZ"]
    )
    return {
        "energy": energy,
        "forces": frame["f"],
        "virial": virial,
        "positions": parsed,
    }
