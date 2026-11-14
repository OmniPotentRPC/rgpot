"""ASE Calculator contract for every potential rgpot exposes.

Each case builds RgpotCalculator for one potential and checks it the way
ASE's own calculators are checked: forces and stress against central
finite differences, energy under rigid translation, the check_state
cache, set() and copying, and a BFGS and an NEB run. A case is skipped
only when its potential is not in this build or needs an external engine
that is not installed, and the reason names what is missing.
"""

from __future__ import annotations

import copy
import os
import pickle
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc100, molecule
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

import rgpot
from rgpot.ase import RgpotCalculator

from support import pt7

try:
    from ase.mep import NEB
except ImportError:  # ASE before 3.23 kept NEB in ase.neb
    from ase.neb import NEB  # type: ignore[no-redef]


def numeric_forces(atoms: Atoms, eps: float) -> np.ndarray:
    try:
        from ase.calculators.fd import calculate_numerical_forces
    except ImportError:  # ASE before 3.24
        from ase.calculators.test import numeric_force

        return np.array(
            [
                [numeric_force(atoms, a, i, eps) for i in range(3)]
                for a in range(len(atoms))
            ]
        )
    return calculate_numerical_forces(atoms, eps=eps)


def numeric_stress(atoms: Atoms, eps: float) -> np.ndarray:
    try:
        from ase.calculators.fd import calculate_numerical_stress
    except ImportError:  # ASE before 3.24
        return atoms.calc.calculate_numerical_stress(atoms, d=eps, voigt=True)
    return calculate_numerical_stress(atoms, eps=eps, voigt=True)


# Geometries. Clusters have pbc False. Periodic cells are sheared and
# rattled so every stress component is nonzero and every force differs.


def _distort(atoms: Atoms, seed: int, rattle: float = 0.03) -> Atoms:
    cell = atoms.cell.array.copy()
    side = float(np.cbrt(abs(np.linalg.det(cell))))
    cell[1, 0] += 0.03 * side
    cell[2, 0] -= 0.02 * side
    cell[2, 1] += 0.025 * side
    atoms.set_cell(cell, scale_atoms=True)
    atoms.rattle(stdev=rattle, seed=seed)
    return atoms


def _as_cluster(atoms: Atoms, seed: int, rattle: float = 0.03) -> Atoms:
    out = Atoms(numbers=atoms.numbers, positions=atoms.positions, pbc=False)
    out.center(vacuum=8.0)
    out.rattle(stdev=rattle, seed=seed)
    return out


LJ_A = 2.0 ** (1.0 / 6.0) * np.sqrt(2.0)


def lj7() -> Atoms:
    atoms = pt7(side=1.12, z=18)
    atoms.rattle(stdev=0.02, seed=7)
    return atoms


def lj_fcc() -> Atoms:
    return _distort(bulk("Ar", "fcc", a=LJ_A, cubic=True).repeat(3), seed=1)


def pt_cluster() -> Atoms:
    atoms = pt7(side=2.85, z=78)
    atoms.rattle(stdev=0.03, seed=2)
    return atoms


def pt_fcc() -> Atoms:
    return _distort(bulk("Pt", "fcc", a=3.92, cubic=True).repeat(2), seed=2)


def al_cluster() -> Atoms:
    atoms = pt7(side=2.7, z=13)
    atoms.rattle(stdev=0.03, seed=3)
    return atoms


def al_fcc() -> Atoms:
    return _distort(bulk("Al", "fcc", a=4.05, cubic=True).repeat(4), seed=3)


def si_cluster() -> Atoms:
    return _as_cluster(bulk("Si", "diamond", a=5.431, cubic=True), seed=4)


def si_diamond() -> Atoms:
    return _distort(bulk("Si", "diamond", a=5.431, cubic=True).repeat(2), seed=4)


def sic_cluster() -> Atoms:
    return _as_cluster(bulk("SiC", "zincblende", a=4.36, cubic=True), seed=5)


def sic_zincblende() -> Atoms:
    return _distort(bulk("SiC", "zincblende", a=4.36, cubic=True).repeat(2), seed=5)


def _fe_he(repeat: int) -> Atoms:
    a = 2.87
    atoms = bulk("Fe", "bcc", a=a, cubic=True).repeat(repeat)
    atoms += Atoms("He", positions=[[0.5 * a, 0.5 * a, a]])
    return atoms


def fehe_cluster() -> Atoms:
    return _as_cluster(_fe_he(2), seed=6)


def fehe_bcc() -> Atoms:
    return _distort(_fe_he(4), seed=6)


def _cu_h2(repeat: tuple[int, int, int]) -> Atoms:
    a = 3.61
    atoms = bulk("Cu", "fcc", a=a, cubic=True).repeat(repeat)
    # Two H at octahedral sites, appended after the copper.
    atoms += Atoms("H2", positions=[[0.5 * a, 0.0, 0.0], [0.5 * a, a, a]])
    return atoms


def cuh2_cluster() -> Atoms:
    return _as_cluster(_cu_h2((2, 2, 1)), seed=8, rattle=0.02)


def cuh2_fcc() -> Atoms:
    return _distort(_cu_h2((4, 4, 4)), seed=8, rattle=0.02)


def _water_h(cell: float, pbc: bool) -> Atoms:
    """Two waters and one H, ordered H of each water, the oxygens, the H.

    This is the order the WaterH entry point reads.
    """
    half = 0.5 * np.deg2rad(104.52)
    arm = np.array([0.9572 * np.sin(half), 0.9572 * np.cos(half), 0.0])
    oxygens = np.array([[3.0, 3.0, 3.0], [3.0, 7.0, 3.0]])
    hydrogens = []
    for o in oxygens:
        hydrogens.append(o + arm)
        hydrogens.append(o + arm * [-1.0, 1.0, 1.0])
    positions = np.vstack([hydrogens, oxygens, [[5.0, 5.0, 3.4]]])
    atoms = Atoms("H4O2H", positions=positions, cell=[cell] * 3, pbc=pbc)
    atoms.rattle(stdev=0.01, seed=9)
    return atoms


def water_h_cluster() -> Atoms:
    return _water_h(12.0, False)


def water_h_box() -> Atoms:
    return _distort(_water_h(12.0, True), seed=9, rattle=0.01)


def ethanol() -> Atoms:
    atoms = molecule("CH3CH2OH")
    atoms.center(vacuum=6.0)
    atoms.pbc = False
    atoms.rattle(stdev=0.02, seed=10)
    return atoms


def _swap_first(z_a: int, z_b: int) -> Callable[[Atoms], np.ndarray]:
    def swap(atoms: Atoms) -> np.ndarray:
        numbers = atoms.numbers.copy()
        i = int(np.flatnonzero(numbers == z_a)[0])
        j = int(np.flatnonzero(numbers == z_b)[0])
        numbers[i], numbers[j] = numbers[j], numbers[i]
        return numbers

    return swap


def _all_to(z: int) -> Callable[[Atoms], np.ndarray]:
    return lambda atoms: np.full(len(atoms), z)


# The i-PI driver for IPIPot: LJ with the parameters of LJPot(u0=1,
# psi=1, cutoff=2.2), minimum image, in i-PI's atomic units.
_IPI_DRIVER = '''
import numpy as np
from ipi.pes.dummy import Dummy_driver

__DRIVER_NAME__ = "rgpot_test_lj"
__DRIVER_CLASS__ = "LJDriver"

BOHR = 0.529177210903
HARTREE = 27.211386245988
EPS = 1.0 / HARTREE
SIGMA = 1.0 / BOHR
CUT = 2.2 / BOHR


def pair(r):
    s6 = (SIGMA / r) ** 6
    return 4 * EPS * (s6 * s6 - s6), 4 * EPS * (6 * s6 - 12 * s6 * s6) / r


class LJDriver(Dummy_driver):
    def compute_structure(self, cell, pos):
        x = np.asarray(pos, dtype=float).reshape(-1, 3)
        rows = np.asarray(cell, dtype=float).T
        d = x[None, :, :] - x[:, None, :]
        frac = d @ np.linalg.inv(rows)
        d = (frac - np.round(frac)) @ rows
        i, j = np.triu_indices(len(x), 1)
        r = np.linalg.norm(d[i, j], axis=-1)
        keep = r < CUT
        i, j, r, dij = i[keep], j[keep], r[keep], d[i[keep], j[keep]]
        e, de = pair(r)
        pot = float(np.sum(e - pair(CUT)[0]))
        f_on_j = -(de / r)[:, None] * dij
        force = np.zeros_like(x)
        np.add.at(force, j, f_on_j)
        np.add.at(force, i, -f_on_j)
        vir = np.einsum("ka,kb->ab", dij, f_on_j)
        return pot, force.reshape(np.shape(pos)), vir, ""
'''


def _ipi_driver_bin() -> str | None:
    return shutil.which("i-pi-py_driver")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class Case:
    """One potential and the systems it is checked on.

    ``kind`` and ``params`` build the calculator unless ``ipi`` is set.
    ``periodic`` is None for a free-boundary potential. ``relaxes`` is
    False for a potential with no minimum (dispersion alone collapses),
    which gets a bounded number of optimizer steps instead.
    ``tweak`` is a parameter change for set(). ``new_numbers`` gives
    other valid atomic numbers for the cache check.
    """

    name: str
    kind: str
    cluster: Callable[[], Atoms]
    periodic: Callable[[], Atoms] | None
    missing: Callable[[], str | None]
    params: dict[str, Any] = field(default_factory=dict)
    tweak: dict[str, Any] | None = None
    new_numbers: Callable[[Atoms], np.ndarray] | None = None
    relaxes: bool = True
    ipi: bool = False
    force_atol: float = 1e-5


def _needs(attr: str) -> Callable[[], str | None]:
    return lambda: None if getattr(rgpot, attr, None) else f"{attr} is not in this build"


def _needs_fortran() -> str | None:
    return None if rgpot.has_fortran_pots else "this build has no Fortran potentials"


def _needs_lammps() -> str | None:
    if not os.environ.get("RGPOT_LJ_ENGINE"):
        return (
            "the LAMMPS LJ engine is an external plugin; set RGPOT_LJ_ENGINE "
            "to liblj_engine.so or liblammps_engine.so"
        )
    return None


def _needs_ipi() -> str | None:
    if getattr(rgpot, "IPIPot", None) is None:
        return "IPIPot is not in this build"
    if _ipi_driver_bin() is None:
        return (
            "IPIPot gets its forces from an external i-PI driver; "
            "i-pi-py_driver (pip install ipi) is not on PATH"
        )
    return None


LJ_PARAMS = {"u0": 1.0, "psi": 1.0, "cutoff": 2.5, "switch_width": 0.5}
MD_LJ = {"u0": 1.0, "psi": 1.0, "cutoff": 2.5}
SMOOTH = {"cutoff": 6.0, "switch_width": 1.0}

CASES = [
    Case("lj", "lj", lj7, lj_fcc, lambda: None, LJ_PARAMS, {"u0": 1.5}, _all_to(2)),
    Case(
        "lj_cluster", "lj_cluster", lj7, None, lambda: None, LJ_PARAMS,
        {"psi": 1.05}, _all_to(2),
    ),
    Case(
        "morse", "morse", pt_cluster, pt_fcc, lambda: None,
        {"cutoff": 6.0, "switch_width": 1.0}, {"De": 0.8},
    ),
    Case("eam_al", "eam_al", al_cluster, al_fcc, _needs_fortran),
    Case("sw", "sw", si_cluster, si_diamond, _needs_fortran),
    Case(
        "stillinger_weber_si", "stillinger_weber_si", si_cluster, si_diamond,
        _needs("StillingerWeberSi"),
    ),
    Case("edip", "edip", si_cluster, si_diamond, _needs_fortran),
    Case("lenosky", "lenosky", si_cluster, si_diamond, _needs_fortran),
    Case("tersoff", "tersoff", si_cluster, si_diamond, _needs_fortran),
    Case(
        "tersoff_sic", "tersoff_sic", sic_cluster, sic_zincblende,
        _needs("TersoffSiC"), new_numbers=_swap_first(14, 6),
    ),
    Case(
        "fehe", "fehe", fehe_cluster, fehe_bcc, _needs_fortran,
        new_numbers=_swap_first(26, 2),
    ),
    Case("cuh2", "cuh2", cuh2_cluster, cuh2_fcc, _needs_fortran),
    Case("water_h", "water_h", water_h_cluster, water_h_box, _needs_fortran),
    # D3Pot and D4Pot build a molecular structure: free boundaries only.
    Case(
        "d3", "d3", ethanol, None, _needs("D3Pot"), {},
        {"functional": "b3lyp"}, _swap_first(6, 8), relaxes=False,
    ),
    Case(
        "d4", "d4", ethanol, None, _needs("D4Pot"), {},
        {"functional": "b3lyp"}, _swap_first(6, 8), relaxes=False,
    ),
    Case(
        "expr", "expr", pt_cluster, pt_fcc, _needs("ExprPot"),
        {
            "expression": "0.5*lj + morse",
            "terms": {
                "lj": {"kind": "lj", "u0": 0.1, "psi": 2.5, **SMOOTH},
                "morse": {"kind": "morse", **SMOOTH},
            },
        },
        {"expression": "lj + morse"},
    ),
    Case(
        "gromacs", "gromacs", lj7, lj_fcc, lambda: None, MD_LJ, {"u0": 1.5},
        _all_to(2),
    ),
    Case(
        "openmm", "openmm", lj7, lj_fcc, lambda: None, MD_LJ, {"u0": 1.5},
        _all_to(2),
    ),
    Case(
        "lammps", "lammps", lj7, lj_fcc, _needs_lammps, MD_LJ, {"u0": 1.5},
        _all_to(2),
    ),
    Case("ipi", "ipi", lj7, lj_fcc, _needs_ipi, ipi=True),
]


@pytest.fixture(params=CASES, ids=[case.name for case in CASES])
def case(request: pytest.FixtureRequest) -> Case:
    reason = request.param.missing()
    if reason:
        pytest.skip(reason)
    return request.param


@pytest.fixture
def make_calc(
    case: Case, tmp_path: Path, request: pytest.FixtureRequest
) -> Callable[[], RgpotCalculator]:
    """Build a fresh calculator for ``case``.

    An IPIPot calculator listens on a free port and gets its own i-PI
    driver process, stopped when the test ends.
    """
    if not case.ipi:
        params = dict(case.params)
        if case.kind == "lammps":
            params["engine_path"] = os.environ["RGPOT_LJ_ENGINE"]
        return lambda: RgpotCalculator(case.kind, **params)

    driver = tmp_path / "rgpot_test_lj.py"
    driver.write_text(_IPI_DRIVER)
    drivers: list[subprocess.Popen[bytes]] = []

    def stop() -> None:
        for proc in drivers:
            proc.kill()
            proc.wait()

    request.addfinalizer(stop)

    def make() -> RgpotCalculator:
        port = _free_port()
        pot = rgpot.IPIPot(address=f"tcp:127.0.0.1:{port}", timeout_s=60.0)
        cmd = [_ipi_driver_bin() or "", "-a", "127.0.0.1", "-p", str(port)]
        cmd += ["-m", "custom", "-P", str(driver)]
        drivers.append(
            subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        )
        return RgpotCalculator(pot)

    return make


def _systems(case: Case) -> list[Atoms]:
    out = [case.cluster()]
    if case.periodic is not None:
        out.append(case.periodic())
    return out


def _counter(calc: RgpotCalculator) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    original = calc.calculate

    def counted(atoms=None, properties=("energy",), system_changes=()):
        calls.append(tuple(system_changes))
        return original(atoms, properties, system_changes)

    calc.calculate = counted  # type: ignore[method-assign]
    return calls


def test_declared_properties(case: Case, make_calc) -> None:
    calc = make_calc()
    props = calc.implemented_properties
    assert props[:3] == ["energy", "free_energy", "forces"]
    assert set(props) <= {"energy", "free_energy", "forces", "stress", "energies"}
    # No kernel reports per-atom energies, so none may claim them.
    assert "energies" not in props
    atoms = case.cluster()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert atoms.get_potential_energy(force_consistent=True) == energy
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert forces.dtype == np.float64
    # A pair-additive or many-body kernel without external field has no
    # net force on an isolated cluster.
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-8)
    for name in ("magmom", "magmoms", "dipole", "charges", "energies"):
        with pytest.raises(PropertyNotImplementedError):
            calc.get_property(name, atoms)
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_potential_energies()
    with pytest.raises(PropertyNotImplementedError):
        calc.get_hessian(atoms)


def test_forces_match_finite_differences(case: Case, make_calc) -> None:
    for atoms in _systems(case):
        atoms.calc = make_calc()
        analytic = atoms.get_forces()
        numeric = numeric_forces(atoms, eps=1e-5)
        scale = max(1.0, float(np.abs(analytic).max()))
        np.testing.assert_allclose(
            analytic, numeric, atol=case.force_atol * scale, rtol=0.0,
            err_msg=f"pbc={atoms.pbc.tolist()}",
        )


def test_stress_matches_finite_differences(case: Case, make_calc) -> None:
    if case.periodic is None:
        atoms = case.cluster()
        atoms.calc = make_calc()
        assert "stress" not in atoms.calc.implemented_properties
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stress()
        return
    atoms = case.periodic()
    atoms.calc = make_calc()
    if "stress" not in atoms.calc.implemented_properties:
        with pytest.raises(PropertyNotImplementedError):
            atoms.get_stress()
        return
    analytic = atoms.get_stress()
    assert analytic.shape == (6,)
    full = atoms.get_stress(voigt=False)
    np.testing.assert_allclose(full, full.T, atol=1e-12)
    numeric = numeric_stress(atoms, eps=1e-6)
    scale = max(1e-3, float(np.abs(analytic).max()))
    np.testing.assert_allclose(analytic, numeric, atol=1e-6 * scale, rtol=0.0)
    # Without a cell there is no volume, so ASE has no stress to compare.
    bare = Atoms(numbers=atoms.numbers, positions=atoms.positions, pbc=False)
    bare.calc = make_calc()
    with pytest.raises(PropertyNotImplementedError):
        bare.get_stress()


def test_lj_stress_sign_is_ase_convention() -> None:
    """A compressed crystal pushes outwards: negative diagonal stress.

    ASE's stress is (1/V) dE/dstrain. Squeezing the LJ crystal 5 percent
    below its lattice constant makes dE/dstrain negative on each axis.
    """
    atoms = bulk("Ar", "fcc", a=0.95 * LJ_A, cubic=True).repeat(3)
    atoms.calc = RgpotCalculator("lj", **LJ_PARAMS)
    stress = atoms.get_stress()
    assert np.all(stress[:3] < 0.0)
    np.testing.assert_allclose(stress[3:], 0.0, atol=1e-10)
    pressure = -stress[:3].mean()
    volume = atoms.get_volume()
    eps = 1e-5
    energies = []
    for factor in (1.0 + eps, 1.0 - eps):
        strained = atoms.copy()
        strained.set_cell(atoms.cell * factor, scale_atoms=True)
        strained.calc = RgpotCalculator("lj", **LJ_PARAMS)
        energies.append(strained.get_potential_energy())
    # P = -dE/dV, in eV/Angstrom^3.
    d_volume = volume * ((1.0 + eps) ** 3 - (1.0 - eps) ** 3)
    assert pressure == pytest.approx(-(energies[0] - energies[1]) / d_volume, rel=1e-6)


def test_energy_is_invariant_under_translation(case: Case, make_calc) -> None:
    for atoms in _systems(case):
        calc = make_calc()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        moved = atoms.copy()
        moved.translate([3.7, -12.2, 25.1])
        if moved.pbc.all():
            moved.wrap()
        moved.calc = calc
        assert moved.get_potential_energy() == pytest.approx(
            energy, rel=1e-10, abs=1e-10
        )
        np.testing.assert_allclose(moved.get_forces(), forces, atol=1e-8)


def test_results_are_cached_until_the_system_changes(case: Case, make_calc) -> None:
    calc = make_calc()
    calls = _counter(calc)
    atoms = case.periodic() if case.periodic is not None else case.cluster()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_potential_energy()
    atoms.get_potential_energy(force_consistent=True)
    if "stress" in calc.implemented_properties:
        atoms.get_stress()
    assert len(calls) == 1

    same = atoms.copy()
    assert calc.check_state(same) == []
    assert calc.get_property("energy", same) == energy
    assert len(calls) == 1

    atoms.positions[0, 0] += 0.01
    assert calc.get_property("energy", atoms, allow_calculation=False) is None
    assert len(calls) == 1
    atoms.get_potential_energy()
    assert len(calls) == 2 and "positions" in calls[-1]

    atoms.set_cell(atoms.cell * 1.01, scale_atoms=False)
    atoms.get_forces()
    assert len(calls) == 3 and "cell" in calls[-1]

    numbers = atoms.copy()
    numbers.numbers = (numbers.numbers % 100) + 1
    assert "numbers" in calc.check_state(numbers)
    if case.new_numbers is not None:
        atoms.numbers = case.new_numbers(atoms)
        atoms.get_potential_energy()
        assert len(calls) == 4 and "numbers" in calls[-1]

    before = len(calls)
    if case.periodic is not None:
        atoms.pbc = False
        atoms.get_potential_energy()
        assert len(calls) == before + 1 and "pbc" in calls[-1]
    else:
        atoms.pbc = True
        assert "pbc" in calc.check_state(atoms)
        with pytest.raises(ValueError, match="free boundaries"):
            atoms.get_potential_energy()


def test_set_returns_changed_keys_and_resets(case: Case, make_calc) -> None:
    calc = make_calc()
    atoms = case.cluster()
    atoms.calc = calc
    first = atoms.get_potential_energy()
    assert calc.set() == {}
    if case.tweak is None:
        assert dict(calc.parameters) == {}
        with pytest.raises(ValueError):
            calc.set(not_a_parameter=1.0)
        assert "energy" in calc.results
        return
    same = {key: calc.parameters[key] for key in case.tweak}
    assert calc.set(**same) == {}
    assert "energy" in calc.results
    with pytest.raises(ValueError, match="unknown"):
        calc.set(not_a_parameter=1.0)
    assert calc.set(**case.tweak) == case.tweak
    assert calc.results == {}
    for key, value in case.tweak.items():
        assert calc.parameters[key] == value
    changed = atoms.get_potential_energy()
    assert changed != first
    fresh = case.cluster()
    fresh.calc = RgpotCalculator(case.kind, **dict(calc.parameters))
    assert fresh.get_potential_energy() == pytest.approx(changed, rel=1e-12)
    assert calc.todict() == {
        key: value
        for key, value in calc.parameters.items()
        if calc.default_parameters.get(key, object()) != value
    }


def test_copy_and_pickle_rebuild_the_potential(case: Case, make_calc) -> None:
    calc = make_calc()
    atoms = case.cluster()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    if case.ipi:
        with pytest.raises(TypeError, match="config"):
            pickle.dumps(calc)
        return
    twin = copy.deepcopy(atoms)
    assert twin.calc is not calc
    assert twin.calc.results["energy"] == energy
    assert twin.calc.check_state(atoms) == []
    twin.calc.reset()
    # dftd4 is reproducible to the last bit or two, not bit for bit.
    assert twin.get_potential_energy() == pytest.approx(energy, rel=1e-12)
    loaded = pickle.loads(pickle.dumps(calc))
    assert loaded.parameters == calc.parameters
    probe = case.cluster()
    probe.calc = loaded
    assert probe.get_potential_energy() == pytest.approx(energy, rel=1e-12)


def test_bfgs_and_trajectory(case: Case, make_calc, tmp_path: Path) -> None:
    atoms = case.cluster()
    atoms.calc = make_calc()
    start = atoms.get_potential_energy()
    path = tmp_path / "opt.traj"
    opt = BFGS(atoms, logfile=None, trajectory=str(path))
    if case.relaxes:
        assert opt.run(fmax=0.05, steps=500)
        assert np.abs(atoms.get_forces()).max() < 0.05
    else:
        # Dispersion forces start below 0.05 eV/Angstrom.
        opt.run(fmax=1e-4, steps=10)
    final = atoms.get_potential_energy()
    assert final < start
    frames = read(path, index=":")
    assert len(frames) >= 2
    assert frames[0].get_potential_energy() == pytest.approx(start)
    assert frames[-1].get_potential_energy() == pytest.approx(final)
    np.testing.assert_allclose(frames[-1].get_forces(), atoms.get_forces())


def test_neb_runs(case: Case, make_calc) -> None:
    initial = case.cluster()
    initial.calc = make_calc()
    if case.relaxes:
        BFGS(initial, logfile=None).run(fmax=0.05, steps=500)
    final = initial.copy()
    final.positions[0] += [0.2, 0.0, 0.0]
    images = [initial.copy() for _ in range(4)] + [final]
    shared = case.ipi
    if shared:
        calc = make_calc()
        for image in images:
            image.calc = calc
    else:
        for image in images:
            image.calc = make_calc()
    neb = NEB(images, allow_shared_calculator=shared, method="improvedtangent")
    neb.interpolate()
    opt = BFGS(neb, logfile=None)
    if case.relaxes:
        assert opt.run(fmax=0.05, steps=300)
    else:
        opt.run(fmax=1e-4, steps=10)
    energies = [image.get_potential_energy() for image in images]
    assert np.all(np.isfinite(energies))
    assert len({id(image.calc) for image in images}) == (1 if shared else 5)


def test_lj_adatom_hop_neb_barrier() -> None:
    """An adatom hop between two hollows on LJ(100) has a saddle.

    The two hollows are equivalent, so their energies agree, and the
    climbing image sits above both.
    """
    def slab(shift: float) -> Atoms:
        atoms = fcc100("Ar", size=(4, 4, 3), a=LJ_A, vacuum=6.0)
        atoms.pbc = True
        top = atoms.positions[atoms.positions[:, 2].argmax()]
        nn = LJ_A / np.sqrt(2.0)
        hollow = top + [0.5 * nn + shift * nn, 0.5 * nn, 0.0]
        hollow[2] = top[2] + 0.75
        atoms += Atoms("Ar", positions=[hollow])
        bottom = atoms.positions[:, 2] < atoms.positions[:, 2].min() + 0.1
        atoms.set_constraint(FixAtoms(mask=bottom))
        atoms.calc = RgpotCalculator("lj", u0=0.1, psi=1.0, cutoff=2.2, switch_width=0.3)
        BFGS(atoms, logfile=None).run(fmax=0.01, steps=500)
        return atoms

    initial, final = slab(0.0), slab(1.0)
    e_initial = initial.get_potential_energy()
    assert final.get_potential_energy() == pytest.approx(e_initial, abs=1e-6)
    images = [initial] + [initial.copy() for _ in range(5)] + [final]
    for image in images[1:-1]:
        image.calc = RgpotCalculator("lj", **dict(initial.calc.parameters))
    neb = NEB(images, climb=True, method="improvedtangent")
    neb.interpolate()
    assert BFGS(neb, logfile=None).run(fmax=0.02, steps=500)
    barrier = max(image.get_potential_energy() for image in images) - e_initial
    assert 0.0 < barrier < 0.2


@pytest.fixture
def potserv_lj():
    """An RgpotRpcCalculator on a managed potserv running LJPot()."""
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("potserv is an external server; set RGPOT_POTSERV_BIN")
    try:
        import capnp  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("RgpotRpcCalculator needs pycapnp")
    from rgpot.ase import RgpotRpcCalculator

    calc = RgpotRpcCalculator.spawn(binary, potential="LJ")
    yield calc
    calc.close()


def test_rpc_calculator_contract(potserv_lj) -> None:
    calc = potserv_lj
    assert calc.implemented_properties == ["energy", "free_energy", "forces"]
    calls = _counter(calc)
    atoms = lj7()
    atoms.calc = calc
    local = lj7()
    local.calc = RgpotCalculator("lj")
    energy = atoms.get_potential_energy()
    assert energy == pytest.approx(local.get_potential_energy(), rel=1e-12)
    assert atoms.get_potential_energy(force_consistent=True) == energy
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces, local.get_forces(), atol=1e-10)
    assert len(calls) == 1
    np.testing.assert_allclose(forces, numeric_forces(atoms, eps=1e-5), atol=1e-5)
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    with pytest.raises(TypeError, match="connection"):
        copy.deepcopy(calc)
    # pbc False ignores the cell, as for the in-process calculator.
    boxed = lj7()
    boxed.set_cell([4.0, 4.0, 4.0])
    boxed.calc = calc
    assert boxed.get_potential_energy() == pytest.approx(energy, rel=1e-12)
    boxed.pbc = [True, False, True]
    with pytest.raises(ValueError, match="non-periodic axes: y"):
        boxed.get_potential_energy()


def test_rpc_calculator_bfgs_and_neb(potserv_lj) -> None:
    initial = lj7()
    initial.calc = potserv_lj
    assert BFGS(initial, logfile=None).run(fmax=0.05, steps=500)
    final = initial.copy()
    final.positions[0] += [0.2, 0.0, 0.0]
    images = [initial.copy() for _ in range(4)] + [final]
    for image in images:
        image.calc = potserv_lj
    neb = NEB(images, allow_shared_calculator=True, method="improvedtangent")
    neb.interpolate()
    assert BFGS(neb, logfile=None).run(fmax=0.05, steps=300)
