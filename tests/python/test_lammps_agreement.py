"""rgpot's classical kernels against LAMMPS on the same parameters.

Opt-in: the module skips when the ``lammps`` Python module or its shared
library is missing. Set ``RGPOT_REQUIRE_LAMMPS=1`` to turn those skips, and
a failed potential download, into errors (the CI job that installs LAMMPS
sets it). ``RGPOT_LAMMPS_REPORT=<path>`` appends one JSON line of
deviations per case.

Both sides read one parameter set:

- ``lj/cut`` with ``pair_modify shift yes`` and ``morse`` with the same
  shift, from the ``LJPot`` and ``MorsePot`` arguments.
- ``eam/alloy`` reads the NIST setfl files below, downloaded at a pinned URL
  and checked against its SHA-256. ``EAMTablePot`` reads the same file.
- Sutton-Chen platinum runs in LAMMPS as ``eam/fs`` from a table written
  here; ``SuttonChenPot`` evaluates the closed form, and ``EAMTablePot``
  reads that table too.
- ``sw`` and ``tersoff`` read ``data/lammps/Si_1985.sw`` and
  ``data/lammps/SiC_1989_erratum.tersoff``, whose values are checked
  against ``StillingerWeberSi.params()`` and ``TersoffSiC().params()``.

Each cell is rattled and strained so no force or stress component is zero
by symmetry. LAMMPS needs a lower-triangular cell, so a cell goes over as
``H = L Q^T`` (QR of ``H^T``). Positions are ``r Q``, and the LAMMPS forces
and stress are rotated back by ``Q^T``. The rgpot stress is the Cauchy
stress, ``sigma = -W / V`` for the LAMMPS virial W.

The relative deviation of an array is ``max|rgpot - LAMMPS| / max|LAMMPS|``.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

_core = pytest.importorskip("rgpot._core")

DATA = Path(__file__).resolve().parent / "data" / "lammps"
REQUIRE = os.environ.get("RGPOT_REQUIRE_LAMMPS", "") not in ("", "0")


def _unavailable(reason: str):
    if REQUIRE:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)


def _preload_mpich() -> None:
    """The PyPI lammps wheel links libmpi.so.12 from the mpich wheel, which
    lands in <prefix>/lib, outside the wheel's runpath."""
    lib = Path(sys.prefix) / "lib" / "libmpi.so.12"
    if sys.platform.startswith("linux") and lib.is_file():
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


try:
    _preload_mpich()
    import lammps

    _probe = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    LAMMPS_VERSION = _probe.version()
    _missing = [
        s
        for s in (
            "lj/cut",
            "morse",
            "eam/alloy",
            "eam/fs",
            "sw",
            "tersoff",
            "edip/multi",
            "meam/spline",
            "zbl",
        )
        if not _probe.has_style("pair", s)
    ]
    _probe.close()
except (ImportError, OSError) as err:
    _unavailable(f"LAMMPS is not available: {err}")
if _missing:
    _unavailable(f"LAMMPS lacks pair styles {_missing}")

# NIST Interatomic Potentials Repository, https://www.ctcms.nist.gov/potentials
POTENTIALS = {
    # Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter and
    # J. D. Kress, Phys. Rev. B 63, 224106 (2001). Setfl conversion by
    # C. A. Becker, 4 February 2009.
    "Cu01.eam.alloy": (
        "https://www.ctcms.nist.gov/potentials/Download/"
        "2001--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A-et-al--Cu-1/2/"
        "Cu01.eam.alloy",
        "5e374260448eea5c26de59432d75e702c1778cc6999d2527a789e5387ce9dc9e",
    ),
    # Y. Mishin, D. Farkas, M. J. Mehl and D. A. Papaconstantopoulos,
    # Phys. Rev. B 59, 3393 (1999). Setfl conversion by C. A. Becker,
    # 30 December 2008.
    "Al99.eam.alloy": (
        "https://www.ctcms.nist.gov/potentials/Download/"
        "1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al/2/"
        "Al99.eam.alloy",
        "60c8a085be79d273324ab421f5b1447578fef55c1acfc6492c0999f15ee8a284",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def potential_file(pytestconfig) -> Callable[[str], Path]:
    """Path of a pinned potential file. RGPOT_LAMMPS_POTENTIALS names a
    directory that already holds it; otherwise it is downloaded once into
    the pytest cache."""
    preset = os.environ.get("RGPOT_LAMMPS_POTENTIALS")
    cache = Path(preset) if preset else pytestconfig.cache.mkdir("lammps-potentials")

    def get(name: str) -> Path:
        url, digest = POTENTIALS[name]
        path = cache / name
        if not path.is_file() or _sha256(path) != digest:
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    payload = resp.read()
            except OSError as err:
                if REQUIRE:
                    pytest.fail(f"cannot fetch {url}: {err}", pytrace=False)
                pytest.skip(f"cannot fetch {url}: {err}")
            if hashlib.sha256(payload).hexdigest() != digest:
                pytest.fail(f"{url} does not match its pinned SHA-256")
            path.write_bytes(payload)
        return path

    return get


# --------------------------------------------------------------------------
# Geometry


FCC = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
DIAMOND = np.vstack([FCC, FCC + 0.25])


def crystal(basis, a, reps):
    """Fractional coordinates of a conventional cell repeated reps times,
    and the cubic cell."""
    reps = np.asarray(reps)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in reps], indexing="ij"), -1)
    frac = (grid.reshape(-1, 1, 3) + basis[None]).reshape(-1, 3) / reps
    return frac, np.diag(a * reps.astype(float))


@dataclass(frozen=True)
class CellCase:
    name: str
    reps: tuple[int, int, int]
    # Applied to the cubic cell H0 as H = H0 @ (I + strain), so a strain
    # with nonzero upper entries gives a cell that LAMMPS must rotate.
    strain: tuple[tuple[float, float, float], ...]
    rattle: float = 0.08
    seed: int = 0

    def build(self, basis, a):
        frac, cell = crystal(basis, a, self.reps)
        cell = cell @ (np.eye(3) + np.asarray(self.strain))
        rng = np.random.default_rng(self.seed)
        pos = frac @ cell + rng.normal(scale=self.rattle, size=frac.shape)
        return pos, cell


ORTHO = ((0.012, 0, 0), (0, -0.008, 0), (0, 0, 0.004))
SYMMETRIC = ((0.01, 0.06, -0.04), (0.06, -0.012, 0.05), (-0.04, 0.05, 0.006))
LOWER = ((0.01, 0, 0), (0.18, -0.01, 0), (-0.12, 0.21, 0.005))
# Thin along the first axis, and a general triclinic cell that is thin too.
THIN_GENERAL = ((0.0, 0.3, 0.1), (0.0, 0.004, -0.2), (0.25, 0.0, 0.0))


def cells(normal, large):
    """normal repetitions keep every face spacing above twice the cutoff.
    large is the repetition of the rattled supercell of about 4000 atoms."""
    return [
        CellCase("orthorhombic", normal, ORTHO, seed=1),
        CellCase("triclinic-symmetric", normal, SYMMETRIC, seed=2),
        CellCase("triclinic-lower", normal, LOWER, seed=3),
        CellCase("thin-orthorhombic", (1, normal[1], normal[2]), ORTHO, seed=4),
        CellCase("thin-triclinic", (1, 2, normal[2]), THIN_GENERAL, seed=5),
        CellCase("large-triclinic", large, LOWER, rattle=0.1, seed=6),
    ]


def lower_triangular(cell):
    """L lower triangular with a positive diagonal and Q orthogonal with
    cell = L @ Q.T."""
    q, r = np.linalg.qr(cell.T)
    sign = np.sign(np.diag(r))
    q = q * sign
    r = sign[:, None] * r
    return r.T, q


# --------------------------------------------------------------------------
# LAMMPS


@dataclass
class Result:
    energy: float
    forces: np.ndarray
    stress: np.ndarray | None
    energies: np.ndarray | None


def run_lammps(pair, positions, types, cell, masses) -> Result:
    """pair is a list of pair_style / pair_coeff commands. types are LAMMPS
    types starting at 1. masses has one entry per type."""
    lower, q = lower_triangular(cell)
    n = len(positions)
    x = positions @ q
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_list(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                "atom_modify map array sort 0 0.0",
                "region box prism 0 {!r} 0 {!r} 0 {!r} {!r} {!r} {!r} units box".format(
                    *(
                        float(lower[i, j])
                        for i, j in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))
                    )
                ),
                f"create_box {len(masses)} box",
            ]
        )
        created = lmp.create_atoms(
            n,
            list(range(1, n + 1)),
            [int(t) for t in types],
            [float(v) for v in x.ravel()],
        )
        assert created == n, f"LAMMPS created {created} of {n} atoms"
        lmp.commands_list(
            [f"mass {k + 1} {m!r}" for k, m in enumerate(masses)]
            + pair
            + [
                "neighbor 1.0 bin",
                "compute pea all pe/atom",
                "compute pesum all reduce sum c_pea",
                "compute vir all pressure NULL virial",
                "thermo_style custom step pe c_pesum c_vir[1]",
                "run 0",
            ]
        )
        ids = np.array(lmp.numpy.extract_atom("id")[:n])
        order = np.argsort(ids)
        f = np.array(lmp.numpy.extract_atom("f")[:n])[order]
        pea = np.array(
            lmp.numpy.extract_compute(
                "pea", lammps.LMP_STYLE_ATOM, lammps.LMP_TYPE_VECTOR
            )[:n]
        )[order]
        energy = lmp.get_thermo("pe")
        # A pointer into LAMMPS: copy it before close().
        vec = lmp.extract_compute("vir", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_VECTOR)
        p = [float(vec[k]) for k in range(6)]
        nktv2p = lmp.extract_global("nktv2p")
    finally:
        lmp.close()
    # Pressure order xx yy zz xy xz yz. The virial pressure is W / V.
    p_lmp = np.array(
        [[p[0], p[3], p[4]], [p[3], p[1], p[5]], [p[4], p[5], p[2]]]
    )
    stress = q @ (-p_lmp / nktv2p) @ q.T
    return Result(energy, f @ q.T, stress, pea)


# --------------------------------------------------------------------------
# Kernels


@dataclass
class Kernel:
    name: str
    basis: np.ndarray
    a: float
    normal: tuple[int, int, int]
    large: tuple[int, int, int]
    # (fixture getter) -> (rgpot potential, pair commands, masses, z -> type)
    setup: Callable
    # Atomic numbers for an n-site cell.
    species: Callable[[int, np.random.Generator], np.ndarray]
    # "atoms" compares per-atom energies, "sum" only their sums, None neither.
    per_atom: str | None = "atoms"
    # Relative tolerance for every quantity not named in the dict.
    tol: float | dict = 1e-10
    tol_reason: str = "analytic on both sides"
    # Why the kernel writes no stress, when it does not.
    no_stress: str | None = None
    # LAMMPS measures the energy from the isolated atom, rgpot does not.
    isolated_atom_zero: bool = False
    skip: str | None = None


def uniform(z):
    return lambda n, rng: np.full(n, z, dtype=np.int32)


def lj_setup(get):
    u0, psi, cut = 0.0104, 3.40, 8.5
    pot = _core.LJPot(u0=u0, cutoff=cut, psi=psi)
    pair = [
        f"pair_style lj/cut {cut!r}",
        f"pair_coeff * * {u0!r} {psi!r}",
        "pair_modify shift yes",
    ]
    return pot, pair, [39.948], {18: 1}


def morse_setup(get):
    de, alpha, re, cut = 0.7102, 1.6047, 2.897, 9.5
    pot = _core.MorsePot(De=de, a=alpha, re=re, cutoff=cut)
    pair = [
        f"pair_style morse {cut!r}",
        f"pair_coeff * * {de!r} {alpha!r} {re!r}",
        "pair_modify shift yes",
    ]
    return pot, pair, [195.084], {78: 1}


def eam_setup(name, element, mass, z):
    def setup(get):
        path = get(name)
        pot = _core.EAMTablePot(str(path), "setfl", periodic=True, stress=True)
        pair = ["pair_style eam/alloy", f"pair_coeff * * {path} {element}"]
        return pot, pair, [mass], {z: 1}

    return setup


# Sutton-Chen platinum, Cagin et al., arXiv:cond-mat/9709114, Table I.
SC = dict(epsilon=1.98330e-2, c=34.408, a=3.92, n=10, m=8)
SC_CUT, SC_RHOMAX, SC_POINTS = 6.5, 1000.0, 50001


def write_sutton_chen_fs(path: Path, points: int = SC_POINTS) -> None:
    """The Sutton-Chen platinum row as a one-element eam/fs table: F(rho) =
    -epsilon c sqrt(rho), rho(r) = (a / r)^m and r phi(r) = r epsilon
    (a / r)^n. Table point 0 sits at r = 0 and is never inside a pair."""
    dr, drho = SC_CUT / (points - 1), SC_RHOMAX / (points - 1)
    r = np.maximum(np.arange(points) * dr, 1.0)
    rho = np.arange(points) * drho
    eps, c, a, n, m = (SC[k] for k in ("epsilon", "c", "a", "n", "m"))
    columns = (-eps * c * np.sqrt(rho), (a / r) ** m, r * eps * (a / r) ** n)
    with open(path, "w") as out:
        out.write("Sutton-Chen platinum\nn 10 m 8\nrgpot test_lammps_agreement\n1 Pt\n")
        out.write(f"{points} {drho!r} {points} {dr!r} {SC_CUT!r}\n78 195.084 3.92 fcc\n")
        for column in columns:
            out.write("".join(f"{v!r}\n" for v in column.tolist()))


@pytest.fixture(scope="session")
def sutton_chen_table(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("sutton-chen") / "Pt_sc.eam.fs"
    write_sutton_chen_fs(path)
    return path


def sc_setup(table_from_file):
    def setup(get):
        path = get("sutton-chen")
        if table_from_file:
            pot = _core.EAMTablePot(str(path), "fs", periodic=True, stress=True)
        else:
            pot = _core.SuttonChenPot(
                **SC, cutoff=SC_CUT, periodic=True, stress=True
            )
        pair = ["pair_style eam/fs", f"pair_coeff * * {path} Pt"]
        return pot, pair, [195.084], {78: 1}

    return setup


def sw_setup(get):
    pot = _core.StillingerWeberSi()
    pair = ["pair_style sw", f"pair_coeff * * {DATA / 'Si_1985.sw'} Si"]
    return pot, pair, [28.0855], {14: 1}


# eOn's SWFortran.f90 epsilon, the one the Fortran SWPot (PotType::SWSi)
# evaluates: EPSILON = 2.16826D0 * 1.06767385d0.
EON_SW_EPSILON = 2.16826 * 1.06767385


@pytest.fixture(scope="session")
def eon_sw_file(tmp_path_factory) -> Path:
    """Si_1985.sw with epsilon replaced by eOn's."""
    (entry,) = _entries(DATA / "Si_1985.sw", 14)
    entry[3] = repr(EON_SW_EPSILON)
    path = tmp_path_factory.mktemp("eon-sw") / "Si_eon.sw"
    path.write_text(" ".join(entry) + "\n")
    return path


def fortran_sw_setup(get):
    pot = _core.SWPot()
    pair = ["pair_style sw", f"pair_coeff * * {get('eon-sw')} Si"]
    return pot, pair, [28.0855], {14: 1}


def tersoff_setup(get):
    pot = _core.TersoffSiC()
    pair = [
        "pair_style tersoff",
        f"pair_coeff * * {DATA / 'SiC_1989_erratum.tersoff'} C Si",
    ]
    return pot, pair, [12.011, 28.0855], {6: 1, 14: 2}


def fortran_tersoff_setup(get):
    _, pair, masses, type_of = tersoff_setup(get)
    return _core.TersoffPot(), pair, masses, type_of


def fortran_edip_setup(get):
    pair = ["pair_style edip/multi", f"pair_coeff * * {DATA / 'Si_eon.edip'} Si"]
    return _core.EDIPPot(), pair, [28.0855], {14: 1}


def fortran_lenosky_setup(get):
    pair = ["pair_style meam/spline", f"pair_coeff * * {DATA / 'Si_eon.meam.spline'} Si"]
    return _core.LenoskyPot(), pair, [28.0855], {14: 1}


def zincblende(n, rng):
    """Silicon on the first fcc sublattice of DIAMOND, carbon on the second."""
    return np.tile(np.repeat(np.array([14, 6], dtype=np.int32), 4), n // 8)


def random_sic(n, rng):
    return rng.choice(np.array([6, 14], dtype=np.int32), size=n)


# Sutton-Chen against its table. Inside the table the PairEAM cubic has
# five-point slopes, so values are O(h^4) and slopes O(h^3), and the gap
# reaches rounding. The closed form is cut at r = 6.5 Angstrom without a
# shift, so the table ends on a nonzero value, and the last interval takes
# its slope from the one-sided difference (f_n - f_n-1) / h, which is
# O(h). A rattled cell always has pairs there, so the largest force gap
# halves with h: 7.5e-9, 3.5e-9, 1.5e-9 and 5.1e-10 at 12501, 25001,
# 50001 and 100001 points on the 4000-atom cell, against 1.8e-10, 1.8e-11,
# 4.7e-12 and 1.0e-11 for the atoms with no partner in the last two
# intervals. Energies, per-atom energies and stress stay under 1e-10.
SC_TOL = {"forces": 1e-8}
SC_TOL_REASON = (
    "closed form against a 50001-point eam/fs table, whose last interval "
    "has a first-order slope at the unshifted cutoff"
)

KERNELS = [
    Kernel("lj", FCC, 5.26, (4, 4, 4), (10, 10, 10), lj_setup, uniform(18)),
    Kernel("morse", FCC, 3.92, (5, 5, 5), (10, 10, 10), morse_setup, uniform(78)),
    Kernel(
        "eam-cu-mishin2001",
        FCC,
        3.615,
        (4, 4, 4),
        (10, 10, 10),
        eam_setup("Cu01.eam.alloy", "Cu", 63.546, 29),
        uniform(29),
        tol_reason=(
            "tabulated: both sides evaluate the same PairEAM cubic of the same "
            "file, so only rounding separates them"
        ),
    ),
    Kernel(
        "eam-al-mishin1999",
        FCC,
        4.05,
        (4, 4, 4),
        (10, 10, 10),
        eam_setup("Al99.eam.alloy", "Al", 26.982, 13),
        uniform(13),
        tol_reason=(
            "tabulated: both sides evaluate the same PairEAM cubic of the same "
            "file, so only rounding separates them"
        ),
    ),
    Kernel(
        "sutton-chen",
        FCC,
        3.92,
        (4, 4, 4),
        (10, 10, 10),
        sc_setup(table_from_file=False),
        uniform(78),
        tol=SC_TOL,
        tol_reason=SC_TOL_REASON,
    ),
    Kernel(
        "sutton-chen-fs-table",
        FCC,
        3.92,
        (4, 4, 4),
        (10, 10, 10),
        sc_setup(table_from_file=True),
        uniform(78),
        tol_reason="tabulated: the same eam/fs table and cubic on both sides",
    ),
    Kernel(
        "sw-si",
        DIAMOND,
        5.431,
        (3, 3, 3),
        (8, 8, 8),
        sw_setup,
        uniform(14),
        per_atom="sum",
    ),
    Kernel(
        "tersoff-si",
        DIAMOND,
        5.432,
        (3, 3, 3),
        (8, 8, 8),
        tersoff_setup,
        uniform(14),
        per_atom="sum",
    ),
    Kernel(
        "fortran-sw-si",
        DIAMOND,
        5.431,
        (3, 3, 3),
        (8, 8, 8),
        fortran_sw_setup,
        uniform(14),
        per_atom=None,
        skip=None if _core.has_fortran_pots else "built without the Fortran kernels",
    ),
    Kernel(
        "fortran-tersoff-si",
        DIAMOND,
        5.432,
        (3, 3, 3),
        (8, 8, 8),
        fortran_tersoff_setup,
        uniform(14),
        per_atom=None,
        no_stress="the Fortran silicon gather writes no virial",
        skip=None if _core.has_fortran_pots else "built without the Fortran kernels",
    ),
    Kernel(
        "fortran-edip-si",
        DIAMOND,
        5.431,
        (3, 3, 3),
        (8, 8, 8),
        fortran_edip_setup,
        uniform(14),
        per_atom=None,
        no_stress="the Fortran silicon gather writes no virial",
        skip=None if _core.has_fortran_pots else "built without the Fortran kernels",
    ),
    Kernel(
        "fortran-lenosky-si",
        DIAMOND,
        5.431,
        (3, 3, 3),
        (8, 8, 8),
        fortran_lenosky_setup,
        uniform(14),
        per_atom=None,
        no_stress="the Fortran silicon gather writes no virial",
        isolated_atom_zero=True,
        skip=None if _core.has_fortran_pots else "built without the Fortran kernels",
    ),
    Kernel(
        "tersoff-c",
        DIAMOND,
        3.5656,
        (4, 4, 4),
        (8, 8, 8),
        tersoff_setup,
        uniform(6),
        per_atom="sum",
    ),
    Kernel(
        "tersoff-sic-zincblende",
        DIAMOND,
        4.36,
        (3, 3, 3),
        (8, 8, 8),
        tersoff_setup,
        zincblende,
        per_atom="sum",
    ),
    Kernel(
        "tersoff-sic-random",
        DIAMOND,
        4.9,
        (3, 3, 3),
        (8, 8, 8),
        tersoff_setup,
        random_sic,
        per_atom="sum",
    ),
]

CASES = [
    pytest.param(k, c, id=f"{k.name}-{c.name}")
    for k in KERNELS
    for c in cells(k.normal, k.large)
]


def rel(got, ref):
    got, ref = np.asarray(got), np.asarray(ref)
    scale = np.max(np.abs(ref))
    return float(np.max(np.abs(got - ref)) / scale) if scale > 0 else float(
        np.max(np.abs(got))
    )


def report(row: dict) -> None:
    path = os.environ.get("RGPOT_LAMMPS_REPORT")
    if path:
        with open(path, "a") as out:
            out.write(json.dumps(row) + "\n")


@pytest.fixture(scope="session")
def getter(potential_file, sutton_chen_table, eon_sw_file):
    def get(name):
        if name == "sutton-chen":
            return sutton_chen_table
        if name == "eon-sw":
            return eon_sw_file
        return potential_file(name)

    return get


@pytest.mark.parametrize(("kernel", "cell_case"), CASES)
def test_agrees_with_lammps(kernel: Kernel, cell_case: CellCase, getter):
    if kernel.skip:
        pytest.skip(kernel.skip)
    pot, pair, masses, type_of = kernel.setup(getter)
    pos, cell = cell_case.build(kernel.basis, kernel.a)
    rng = np.random.default_rng(cell_case.seed + 100)
    z = kernel.species(len(pos), rng)
    types = np.array([type_of[int(v)] for v in z])

    want_atoms = kernel.per_atom is not None and pot.caps()["energies"]
    got = pot.evaluate(pos, z, cell, energies=want_atoms)
    ref = run_lammps(pair, pos, types, cell, masses)
    if kernel.isolated_atom_zero:
        lone = {
            int(v): pot.evaluate(np.zeros((1, 3)), [int(v)], 30.0 * np.eye(3))["energy"]
            for v in np.unique(z)
        }
        got["energy"] -= sum(lone[int(v)] for v in z)

    dev = {
        "energy": abs(got["energy"] - ref.energy) / abs(ref.energy),
        "forces": rel(got["forces"], ref.forces),
    }
    if kernel.no_stress:
        assert got["stress"] is None
    else:
        assert got["stress"] is not None, f"{kernel.name} wrote no stress"
        stress = np.asarray(got["stress"])
        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        dev["stress"] = rel(
            [stress[i, j] for i, j in voigt], [ref.stress[i, j] for i, j in voigt]
        )
        asym = float(np.max(np.abs(stress - stress.T)))
        assert asym <= 1e-12 * max(1.0, float(np.max(np.abs(stress))))
    assert np.isclose(ref.energies.sum(), ref.energy, rtol=1e-12, atol=0)
    if want_atoms:
        e = np.asarray(got["energies"])
        assert np.isclose(e.sum(), got["energy"], rtol=1e-12, atol=0)
        if kernel.per_atom == "atoms":
            dev["per_atom"] = rel(e, ref.energies)
        else:
            dev["per_atom_sum"] = abs(e.sum() - ref.energies.sum()) / abs(
                ref.energies.sum()
            )
    report(
        {
            "kernel": kernel.name,
            "cell": cell_case.name,
            "atoms": len(pos),
            "lammps": LAMMPS_VERSION,
            **dev,
        }
    )
    tols = kernel.tol if isinstance(kernel.tol, dict) else {}
    default = kernel.tol if not isinstance(kernel.tol, dict) else 1e-10
    bad = {
        k: (v, tols.get(k, default))
        for k, v in dev.items()
        if not v <= tols.get(k, default)
    }
    assert not bad, (
        f"{kernel.name} on {cell_case.name} ({len(pos)} atoms): "
        f"(deviation, tolerance) {bad} ({kernel.tol_reason})"
    )


# ZBL has free boundaries, so it is compared on rattled, compressed fcc
# clusters of mixed species centred in a cubic box. The vacuum exceeds
# twice the cutoff, so LAMMPS's periodic images never interact, and both
# sides take the stress over the box volume.
ZBL_SPECIES = {6: 12.011, 14: 28.0855, 28: 58.6934}


@pytest.mark.parametrize("reps", [(3, 3, 3), (10, 10, 10)], ids=["108", "4000"])
def test_zbl_cluster_agrees_with_lammps(reps):
    cut_inner, cut_global = 2.0, 2.5
    pot = _core.ZBLPot(cut_inner=cut_inner, cut_global=cut_global)
    frac, cell = crystal(FCC, 2.9, reps)
    rng = np.random.default_rng(7)
    pos = frac @ cell + rng.normal(scale=0.12, size=frac.shape)
    side = float(np.ptp(pos, axis=0).max()) + 3.0 * cut_global
    pos += side / 2 - pos.mean(axis=0)
    cell = side * np.eye(3)
    zs = sorted(ZBL_SPECIES)
    z = rng.choice(np.array(zs, dtype=np.int32), size=len(pos))
    types = np.array([zs.index(int(v)) + 1 for v in z])
    pair = [f"pair_style zbl {cut_inner!r} {cut_global!r}"] + [
        f"pair_coeff {i + 1} {j + 1} {zs[i]} {zs[j]}"
        for i in range(len(zs))
        for j in range(i, len(zs))
    ]

    got = pot.evaluate(pos, z, cell)
    ref = run_lammps(pair, pos, types, cell, [ZBL_SPECIES[v] for v in zs])
    voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    stress = np.asarray(got["stress"])
    dev = {
        "energy": abs(got["energy"] - ref.energy) / abs(ref.energy),
        "forces": rel(got["forces"], ref.forces),
        "stress": rel(
            [stress[i, j] for i, j in voigt], [ref.stress[i, j] for i, j in voigt]
        ),
    }
    report(
        {
            "kernel": "zbl",
            "cell": "cluster",
            "atoms": len(pos),
            "lammps": LAMMPS_VERSION,
            **dev,
        }
    )
    assert all(v <= 1e-10 for v in dev.values()), dev


# --------------------------------------------------------------------------
# The tracked parameter files are the kernels' constants.


def _entries(path: Path, width: int):
    words = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].split()
        words.extend(line)
    assert len(words) % width == 0
    return [words[k : k + width] for k in range(0, len(words), width)]


def test_sw_file_is_the_kernel():
    (entry,) = _entries(DATA / "Si_1985.sw", 14)
    assert entry[:3] == ["Si"] * 3
    names = ["epsilon", "sigma", "a", "lambda", "gamma", "costheta0", "A", "B", "p", "q"]
    values = dict(zip(names, map(float, entry[3:13])))
    assert values == _core.StillingerWeberSi.params()
    assert float(entry[13]) == 0.0


def test_tersoff_file_is_the_kernel():
    params = _core.TersoffSiC().params()
    entries = _entries(DATA / "SiC_1989_erratum.tersoff", 17)
    assert len(entries) == 8
    for row in entries:
        i, j, k = row[:3]
        m, gamma, lam3, c, d, h, n, beta, lam2, big_b, big_r, big_d, lam1, big_a = map(
            float, row[3:]
        )
        s = params["species"][i]
        pij = params["pairs"][(i, j)]
        pik = params["pairs"][(i, k)]
        assert (m, lam3) == (3.0, 0.0)
        assert (gamma, c, d, h, n, beta) == (
            s["gamma"],
            s["c"],
            s["d"],
            s["costheta0"],
            s["n"],
            s["beta"],
        )
        assert (lam1, lam2, big_a, big_b) == (
            pij["lambda1"],
            pij["lambda2"],
            pij["A"],
            pij["B"],
        )
        assert big_r - big_d == pytest.approx(pik["rin"], rel=1e-15)
        assert big_r + big_d == pytest.approx(pik["rout"], rel=1e-15)


FORTRAN = Path(__file__).resolve().parents[2] / "CppCore" / "rgpot" / "fortran"


def _fortran_source(name: str) -> str:
    path = FORTRAN / name
    if not path.exists():
        pytest.skip(f"no {name} in this tree")
    return path.read_text()


def _fortran_array(src: str, name: str) -> list[float]:
    body = re.search(rf"\b{name}\(0:\d+\) = \[(.*?)\]", src, re.S).group(1)
    return [float(v.replace("_wp", "")) for v in body.replace("&", "").split(",")]


def test_edip_file_is_the_kernel():
    src = _fortran_source("rgpot_edip.f90")
    par = {
        m[0]: float(m[1])
        for m in re.findall(r"real\(wp\) :: (\w+) = ([-\d.]+)_wp", src)
    }
    (entry,) = _entries(DATA / "Si_eon.edip", 20)
    assert entry[:3] == ["Si"] * 3
    big_a, big_b, cut_a, cut_c, alpha, beta, eta, gamma, lam, mu, rho, sigma, q0 = map(
        float, entry[3:16]
    )
    assert big_a == pytest.approx(par["cap_a"] * par["palp"], rel=1e-15)
    assert big_a * big_b ** rho == pytest.approx(
        par["cap_a"] * par["cap_b"] ** par["rh"], rel=1e-15
    )
    assert eta == pytest.approx(par["delta"] / par["qo"], rel=1e-15)
    assert cut_a == par["a"] == par["b"]
    assert (cut_c, alpha, beta, gamma, lam, mu, rho, sigma, q0) == tuple(
        par[k] for k in ("c", "alp", "bet", "gam", "lam", "mu", "rh", "sig", "qo")
    )
    assert list(map(float, entry[16:])) == [par[k] for k in ("u1", "u2", "u3", "u4")]


def test_lenosky_file_is_the_kernel():
    src = _fortran_source("rgpot_lenosky.f90")
    lines = (DATA / "Si_eon.meam.spline").read_text().splitlines()[1:]
    for name in ("phi", "rho", "uuu", "fff", "ggg"):
        lo, hi = map(
            float,
            re.search(
                rf"sp_{name} = spline_t\(\s*&?\s*tmin=([-\d.]+)_wp, tmax=([-\d.]+)_wp",
                src,
            ).groups(),
        )
        y, y2 = _fortran_array(src, f"cof_{name}"), _fortran_array(src, f"dof_{name}")
        n = int(lines[0])
        knots = np.array([list(map(float, row.split())) for row in lines[3 : 3 + n]])
        lines = lines[3 + n :]
        assert n == len(y)
        np.testing.assert_allclose(knots[[0, -1], 0], [lo, hi], rtol=1e-15)
        assert knots[:, 1].tolist() == y and knots[:, 2].tolist() == y2


def test_lower_triangular_round_trip():
    cell = np.array([[5.0, 0.4, -0.3], [0.2, 4.6, 0.7], [-0.5, 0.1, 6.1]])
    lower, q = lower_triangular(cell)
    assert np.allclose(np.triu(lower, 1), 0.0)
    assert np.all(np.diag(lower) > 0)
    np.testing.assert_allclose(lower @ q.T, cell, rtol=0, atol=1e-14)
    np.testing.assert_allclose(q.T @ q, np.eye(3), rtol=0, atol=1e-15)
