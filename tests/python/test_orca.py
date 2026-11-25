# Copyright 2023--present rgpot developers
"""ORCAPot over the fake liborcac, called directly and through ASE.

The fake (CppCore/tests/orcac_fake_engine.c) returns
-1 + 0.01 charge + 0.001 (multiplicity - 1) Eh and gradient component k
of 0.001 (k + 1) Eh/bohr, without ORCA. It speaks the frozen orcac ABI 1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

import rgpot
from rgpot.ase import RgpotCalculator

pytestmark = pytest.mark.skipif(
    not hasattr(rgpot, "ORCAPot") or sys.platform == "win32",
    reason="needs ORCAPot and a POSIX C compiler for the fake liborcac",
)

_REPO = Path(__file__).resolve().parents[2]
_FAKE = _REPO / "CppCore" / "tests" / "orcac_fake_engine.c"
_HEADER_DIR = _REPO / "CppCore" / "rgpot" / "ORCAPot"
_EXECUTABLE = "/bin/sh"

HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANGSTROM = HARTREE_TO_EV / 0.529177210903

WATER = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
WATER_Z = np.array([8, 1, 1], dtype=np.int32)


def _compile(out: Path, *defines: str) -> Path:
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if cc is None or not _FAKE.is_file():
        pytest.skip("no C compiler or no fake liborcac source")
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    lib = out.with_suffix(suffix)
    cmd = [cc, "-shared", "-fPIC", "-O1", f"-I{_HEADER_DIR}", str(_FAKE), "-o", str(lib)]
    cmd += [f"-D{d}" for d in defines]
    subprocess.run(cmd, check=True, capture_output=True)
    return lib


@pytest.fixture(scope="module")
def fake(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _compile(tmp_path_factory.mktemp("orcac") / "liborcac_fake")


def _pot(fake: Path, **kwargs: object) -> object:
    kwargs.setdefault("method", "B3LYP")
    kwargs.setdefault("executable", _EXECUTABLE)
    return rgpot.ORCAPot(library=str(fake), **kwargs)


def _fake_energy_ev(charge: int = 0, multiplicity: int = 1) -> float:
    return (-1.0 + 0.01 * charge + 0.001 * (multiplicity - 1)) * HARTREE_TO_EV


def _fake_forces(n_atoms: int) -> np.ndarray:
    k = np.arange(1, 3 * n_atoms + 1, dtype=np.float64)
    return -(0.001 * k * HARTREE_BOHR_TO_EV_ANGSTROM).reshape(n_atoms, 3)


def test_orca_padded_box_equals_no_box(fake: Path) -> None:
    pot = _pot(fake, charge=1, multiplicity=2)
    assert pot.library_path == str(fake)
    assert pot.engine_version == "orcac-fake/1.0"
    assert pot.executable_identity == os.path.realpath(_EXECUTABLE)
    caps = pot.caps()
    assert caps["periodic"] is False
    assert caps["stress"] is False
    zero_e, zero_f, _ = pot(WATER, WATER_Z, np.zeros((3, 3)))
    padded_e, padded_f, _ = pot(WATER, WATER_Z, 100.0 * np.eye(3))
    assert padded_e == pytest.approx(_fake_energy_ev(1, 2), rel=1e-12)
    assert zero_e == padded_e
    np.testing.assert_allclose(padded_f, _fake_forces(3), rtol=1e-12)
    np.testing.assert_array_equal(zero_f, padded_f)


@pytest.mark.parametrize(
    "box",
    [
        np.eye(3),
        # Long cell vectors whose difference brings an image within 1.2 A.
        np.array([[100.0, 0.0, 0.0], [99.5, 1.0, 0.0], [0.0, 0.0, 100.0]]),
    ],
    ids=["small", "skewed"],
)
def test_orca_refuses_a_wrapping_cell(fake: Path, box: np.ndarray) -> None:
    pot = _pot(fake)
    with pytest.raises(ValueError, match="wrap across a periodic boundary"):
        pot(WATER, WATER_Z, box)


def test_orca_config_round_trips(fake: Path) -> None:
    pot = _pot(
        fake,
        basis="def2-SVP",
        blocks="%scf\n  MaxIter 200\nend",
        nprocs=2,
        guess_policy="fresh",
        fresh_retry=False,
    )
    config = pot.config()
    assert config["method"] == "B3LYP"
    assert config["basis"] == "def2-SVP"
    assert config["blocks"] == "%scf\n  MaxIter 200\nend"
    assert config["nprocs"] == 2
    assert config["multiplicity"] == 1
    assert config["guess_policy"] == "fresh"
    assert config["fresh_retry"] is False
    assert config["library"] == str(fake)
    assert _pot(fake).config()["guess_policy"] == "fresh"
    assert _pot(fake, guess_policy="moread").config()["guess_policy"] == "moread"
    with pytest.raises(ValueError, match="guess_policy"):
        _pot(fake, guess_policy="sometimes")
    with pytest.raises(ValueError, match="keep_files"):
        _pot(fake, keep_files=True)


def test_orca_absent_library_raises(tmp_path: Path) -> None:
    missing = tmp_path / "liborcac-missing.so"
    with pytest.raises(RuntimeError, match="liborcac not loaded"):
        rgpot.ORCAPot(method="B3LYP", library=str(missing))


@pytest.mark.parametrize(
    ("define", "reason"),
    [
        ("ORCAC_FAKE_C_ABI=2", "reports C ABI version 2, but rgpot was built against 1"),
        ("ORCAC_FAKE_ABI=2", "reports ABI version 2, but rgpot was built against 1"),
        ("ORCAC_FAKE_SMALL_CONFIG", "OrcaCConfig struct_size"),
        ("ORCAC_FAKE_SMALL_RESULT", "OrcaCResult struct_size"),
        ("ORCAC_FAKE_NO_COMPUTE", "lacks orcac_energy_gradient"),
    ],
)
def test_orca_refuses_a_mismatched_liborcac(
    tmp_path: Path, define: str, reason: str
) -> None:
    lib = _compile(tmp_path / "liborcac_variant", define)
    with pytest.raises(RuntimeError, match=reason):
        rgpot.ORCAPot(method="B3LYP", executable=_EXECUTABLE, library=str(lib))


@pytest.mark.parametrize(
    ("method", "status"),
    [("FAIL", "ORCAC_E_ERROR"), ("NOCONV", "ORCAC_E_NOT_CONVERGED")],
)
def test_orca_failed_run_raises(fake: Path, method: str, status: str) -> None:
    pot = _pot(fake, method=method)
    with pytest.raises(RuntimeError, match=status):
        pot(WATER, WATER_Z, np.zeros((3, 3)))


def test_orca_executable_search(fake: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORCAC_ORCA", raising=False)
    with pytest.raises(RuntimeError, match="ORCAC_E_NOT_FOUND"):
        _pot(fake, executable="")
    monkeypatch.setenv("ORCAC_ORCA", _EXECUTABLE)
    found = _pot(fake, executable="")
    assert found.executable_identity == os.path.realpath(_EXECUTABLE)


def test_orca_pins_the_executable(fake: Path, tmp_path: Path) -> None:
    link = tmp_path / "orca"
    link.symlink_to(_EXECUTABLE)
    pot = _pot(fake, executable=str(link))
    assert pot.executable_identity == os.path.realpath(_EXECUTABLE)
    energy, _, _ = pot(WATER, WATER_Z, np.zeros((3, 3)))
    assert energy == pytest.approx(_fake_energy_ev(), rel=1e-12)
    copy = tmp_path / "orca-copy"
    shutil.copy2(_EXECUTABLE, copy)
    link.unlink()
    link.symlink_to(copy)
    with pytest.raises(RuntimeError, match="changed since the session was opened"):
        pot(WATER, WATER_Z, np.zeros((3, 3)))
    moved = _pot(fake, executable=str(link))
    assert moved.executable_identity == os.path.realpath(copy)


@pytest.mark.parametrize("kind", ["ORCAPot", "orca"])
def test_ase_cluster_in_a_padded_box(fake: Path, kind: str) -> None:
    atoms = Atoms("OH2", positions=WATER, pbc=False)
    kwargs = {"method": "B3LYP", "executable": _EXECUTABLE, "library": str(fake)}
    atoms.calc = RgpotCalculator(kind, **kwargs)
    assert type(atoms.calc.potential).__name__ == "ORCAPot"
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert energy == pytest.approx(_fake_energy_ev(), rel=1e-12)
    np.testing.assert_allclose(forces, _fake_forces(3), rtol=1e-12)

    centred = atoms.copy()
    centred.center(vacuum=6.0)
    centred.calc = RgpotCalculator(kind, **kwargs)
    assert centred.get_potential_energy() == energy
