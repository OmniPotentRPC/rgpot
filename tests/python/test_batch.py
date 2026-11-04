"""calculate_many matches N single evaluations, bit for bit."""

from __future__ import annotations

import numpy as np
import pytest

import rgpot
from rgpot.ase import RgpotCalculator

from support import lj_dimer, pt7, raw_bytes


def _frames():
    frames = [lj_dimer(distance) for distance in (1.2, 1.5, 2.2)]
    if rgpot.has_fortran_pots:
        frames.extend(pt7(jitter=0.02, seed=index) for index in range(3))
    return frames


def _serial(calc: RgpotCalculator, frames) -> list[dict]:
    rows = []
    for atoms in frames:
        row = calc.__rg_evaluate__(
            atoms.get_positions(),
            atoms.get_atomic_numbers(),
            np.asarray(atoms.cell.array),
            atoms.pbc,
        )
        rows.append(row)
    return rows


def test_calculate_many_matches_serial_bytes() -> None:
    frames = _frames()
    lj_frames = [atoms for atoms in frames if int(atoms.numbers[0]) == 1]
    lj = RgpotCalculator("lj")
    batched = lj.calculate_many(lj_frames, threads=1)
    serial = _serial(lj, lj_frames)
    assert len(batched) == len(serial)
    for left, right in zip(batched, serial):
        assert raw_bytes(left["energy"], left["forces"]) == raw_bytes(
            right["energy"], right["forces"]
        )
        assert left["free_energy"] == left["energy"]


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build")
def test_eam_batch_matches_serial_at_several_widths() -> None:
    frames = [pt7(jitter=0.03, seed=index) for index in range(8)]
    calc = RgpotCalculator("eam_al")
    serial = _serial(calc, frames)
    reference = [
        raw_bytes(row["energy"], row["forces"]) for row in serial
    ]
    for threads in (1, 2, 4):
        batched = calc.calculate_many(frames, threads=threads)
        got = [raw_bytes(row["energy"], row["forces"]) for row in batched]
        assert got == reference


def test_calculate_many_does_not_mutate_inputs() -> None:
    frames = [lj_dimer(1.4), lj_dimer(1.8)]
    snapshot = [atoms.get_positions().copy() for atoms in frames]
    numbers = [atoms.get_atomic_numbers().copy() for atoms in frames]
    calc = RgpotCalculator("lj")
    calc.calculate_many(frames, attach=False)
    for atoms, pos, nums in zip(frames, snapshot, numbers):
        np.testing.assert_array_equal(atoms.get_positions(), pos)
        np.testing.assert_array_equal(atoms.get_atomic_numbers(), nums)
        assert atoms.calc is None


def test_attach_writes_single_point() -> None:
    frames = [lj_dimer(1.4), lj_dimer(1.8)]
    calc = RgpotCalculator("lj")
    rows = calc.calculate_many(frames, attach=True, threads=2)
    for atoms, row in zip(frames, rows):
        assert atoms.calc is not None
        assert atoms.get_potential_energy() == row["energy"]
        np.testing.assert_array_equal(atoms.get_forces(), row["forces"])


def test_empty_batch() -> None:
    calc = RgpotCalculator("lj")
    assert calc.calculate_many([]) == []
