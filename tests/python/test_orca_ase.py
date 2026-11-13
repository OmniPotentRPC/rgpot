"""ORCA through the ASE calculator, against ASE's own ORCA reader."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import rgpot
from rgpot.ase import RgpotCalculator

pytest.importorskip("ase")
from ase import Atoms  # noqa: E402

_REPLAY = (
    Path(__file__).resolve().parents[2] / "CppCore" / "tests" / "data" / "orca"
)
_FAKE = r'''#!/usr/bin/python3
import os
import sys
from pathlib import Path

def main() -> int:
    inp = Path(sys.argv[1])
    target = inp if inp.is_absolute() else Path.cwd() / inp
    replay = Path(os.environ["ORCA_FAKE_REPLAY"])
    target.with_suffix(".engrad").write_text(
        (replay / "replay.engrad").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    sys.stdout.write((replay / "replay.out").read_text(encoding="utf-8"))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _water() -> Atoms:
    atoms = Atoms(
        "OHH",
        positions=[
            (0.0, 0.0, 0.1173),
            (0.0, 0.7572, -0.4692),
            (0.0, -0.7572, -0.4692),
        ],
        pbc=False,
    )
    return atoms


def test_periodic_orca_is_rejected() -> None:
    if rgpot.ORCAPot is None:
        pytest.skip("this build has no ORCAPot")
    atoms = _water()
    atoms.set_cell([5.0, 5.0, 5.0])
    atoms.set_pbc(True)
    atoms.calc = RgpotCalculator("orca", executable="/no/such/orca")
    with pytest.raises(ValueError, match="periodic boundary conditions"):
        atoms.get_potential_energy()


def test_recorded_output_matches_ase(tmp_path: Path) -> None:
    if rgpot.ORCAPot is None:
        pytest.skip("this build has no ORCAPot")
    from ase.calculators.orca import ORCA, OrcaProfile

    script = tmp_path / "orca"
    script.write_text(_FAKE, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    os.environ["ORCA_FAKE_REPLAY"] = str(_REPLAY)
    try:
        ours = _water()
        ours.calc = RgpotCalculator("orca", executable=str(script))
        their = _water()
        their.calc = ORCA(
            profile=OrcaProfile(str(script)),
            directory=str(tmp_path / "ase"),
            charge=0,
            mult=1,
            orcasimpleinput="HF def2-SVP",
            orcablocks="",
        )
        assert ours.get_potential_energy() == pytest.approx(
            their.get_potential_energy(), rel=1e-6
        )
        assert ours.get_forces() == pytest.approx(their.get_forces(), rel=1e-6)
    finally:
        os.environ.pop("ORCA_FAKE_REPLAY", None)
