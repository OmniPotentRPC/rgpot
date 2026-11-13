"""ORCA through ASE and potserv, against the recorded replay files.

The fake ``orca`` copies ``replay.engrad`` and prints ``replay.stdout``.
No ORCA installation is required.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest

import rgpot
from rgpot.ase import RgpotCalculator

pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.optimize import FIRE  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_REPLAY = _ROOT / "CppCore" / "tests" / "data" / "orca"
_FAKE = _ROOT / "CppCore" / "tests" / "data" / "orca" / "fake_orca.sh"

# CODATA 2018, the same factors as rgpot/units.hpp.
_HARTREE_TO_EV = 27.211386245988
_BOHR_TO_ANGSTROM = 0.529177210903
_NEG_GRAD = -_HARTREE_TO_EV / _BOHR_TO_ANGSTROM
_RECORDED_HA = -75.5
_RECORDED_GRAD = np.array(
    [
        [0.020, -0.010, 0.005],
        [-0.015, 0.012, -0.003],
        [-0.005, -0.002, -0.002],
    ],
    dtype=np.float64,
)


def _require_orca() -> None:
    if rgpot.OrcaPot is None:
        pytest.skip("this build has no OrcaPot")


def _water() -> Atoms:
    return Atoms(
        "OHH",
        positions=[
            (0.0, 0.0, 0.1173),
            (0.0, 0.7572, -0.4692),
            (0.0, -0.7572, -0.4692),
        ],
        pbc=False,
    )


def _install_fake(tmp_path: Path) -> Path:
    script = tmp_path / "orca"
    shutil.copy(_FAKE, script)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    os.environ["ORCA_FAKE_REPLAY"] = str(_REPLAY)
    os.environ.pop("RGPOT_ORCA", None)
    os.environ.pop("ORCA", None)
    return script


def _expect_recorded(energy: float, forces: np.ndarray) -> None:
    assert energy == pytest.approx(_RECORDED_HA * _HARTREE_TO_EV, rel=1e-10)
    assert forces == pytest.approx(_RECORDED_GRAD * _NEG_GRAD, rel=1e-8, abs=1e-8)


def test_periodic_orca_is_rejected() -> None:
    _require_orca()
    atoms = _water()
    atoms.set_cell([5.0, 5.0, 5.0])
    atoms.set_pbc(True)
    atoms.calc = RgpotCalculator("orca", executable="/no/such/orca")
    with pytest.raises(ValueError, match="periodic boundary conditions"):
        atoms.get_potential_energy()


def test_recorded_output_matches_ase(tmp_path: Path) -> None:
    _require_orca()
    script = _install_fake(tmp_path)
    try:
        ours = _water()
        ours.calc = RgpotCalculator(
            "orca", executable=str(script), basis="def2-SVP"
        )
        energy = float(ours.get_potential_energy())
        forces = np.asarray(ours.get_forces(), dtype=np.float64)
        _expect_recorded(energy, forces)

        try:
            from ase.calculators.orca import ORCA, OrcaProfile
        except ImportError:
            return
        their = _water()
        their.calc = ORCA(
            profile=OrcaProfile(str(script)),
            directory=str(tmp_path / "ase"),
            charge=0,
            mult=1,
            orcasimpleinput="HF def2-SVP",
            orcablocks="",
        )
        assert energy == pytest.approx(their.get_potential_energy(), rel=1e-6)
        assert forces == pytest.approx(their.get_forces(), rel=1e-6)
    finally:
        os.environ.pop("ORCA_FAKE_REPLAY", None)


def test_rpc_bfgs_h2o_replay(tmp_path: Path) -> None:
    """ASE BFGS of H2O through potserv and the replay ``orca``."""
    _require_orca()
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        built = _ROOT / "build" / "CppCore" / "potserv"
        binary = str(built) if built.is_file() else ""
    if not binary:
        pytest.skip("potserv was not built")

    script = _install_fake(tmp_path)
    os.environ["ORCA_EXE"] = str(script)
    from rgpot.ase import RgpotRpcCalculator

    start = _water()
    start_pos = start.positions.copy()
    try:
        calc = RgpotRpcCalculator.spawn(binary, potential="ORCA:HF:def2-SVP")
        try:
            start.calc = calc
            _expect_recorded(
                float(start.get_potential_energy()),
                np.asarray(start.get_forces(), dtype=np.float64),
            )
            # The replay gradient does not change with geometry, so a
            # quasi-Newton step has no curvature. FIRE still walks downhill
            # for a fixed number of steps.
            FIRE(start, logfile=None, trajectory=None).run(fmax=1e-3, steps=5)
            _expect_recorded(
                float(start.get_potential_energy()),
                np.asarray(start.get_forces(), dtype=np.float64),
            )
        finally:
            calc.close()
    finally:
        os.environ.pop("ORCA_FAKE_REPLAY", None)
        os.environ.pop("ORCA_EXE", None)
    assert not np.allclose(start.positions, start_pos)
