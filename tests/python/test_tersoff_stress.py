# MIT License
# Copyright 2023--present rgpot developers
"""Tersoff analytic stress finite-difference and thread-agreement checks."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import rgpot
from rgpot.ase import RgpotCalculator

_STRAIN_STEP = 2e-4
_FD_TOL = 5e-10
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_VOIGT_FROM_FULL = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _silicon_cell(seed: int = 20261124) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = bulk("Si", "diamond", a=5.432, cubic=True).repeat((2, 2, 2))
    rng = np.random.default_rng(seed)
    pos = atoms.positions + rng.uniform(-0.12, 0.12, size=atoms.positions.shape)
    box = atoms.cell.array.copy()
    box[1, 0] += 0.06 * 5.432
    box[2, 0] -= 0.03 * 5.432
    box[2, 1] += 0.04 * 5.432
    return pos, atoms.numbers.astype(np.int32), box


def _sic_cell(seed: int = 20261124) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms = bulk("Si", "diamond", a=4.03, cubic=True).repeat((2, 2, 2))
    numbers = np.where((np.arange(len(atoms)) % 8) < 4, 6, 14).astype(np.int32)
    rng = np.random.default_rng(seed)
    pos = atoms.positions + rng.uniform(-0.04, 0.04, size=atoms.positions.shape)
    box = atoms.cell.array.copy()
    box[1, 0] += 0.18
    box[2, 1] -= 0.11
    return pos, numbers, box


def _strained(
    pos: np.ndarray, box: np.ndarray, voigt: int, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    a, b = _VOIGT_PAIRS[voigt]
    deformation = np.eye(3)
    if a == b:
        deformation[a, a] += eps
    else:
        deformation[a, b] += 0.5 * eps
        deformation[b, a] += 0.5 * eps
    return pos @ deformation.T, box @ deformation.T


def _fd_stress_voigt(eval_energy, pos: np.ndarray, box: np.ndarray) -> np.ndarray:
    vol = abs(float(np.linalg.det(box)))
    out = np.empty(6, dtype=np.float64)
    for v in range(6):
        e_p1 = eval_energy(*_strained(pos, box, v, _STRAIN_STEP))
        e_m1 = eval_energy(*_strained(pos, box, v, -_STRAIN_STEP))
        e_p2 = eval_energy(*_strained(pos, box, v, 2.0 * _STRAIN_STEP))
        e_m2 = eval_energy(*_strained(pos, box, v, -2.0 * _STRAIN_STEP))
        out[v] = (8.0 * (e_p1 - e_m1) - (e_p2 - e_m2)) / (
            12.0 * _STRAIN_STEP * vol
        )
    return out


def _full_to_voigt(stress_3x3: np.ndarray) -> np.ndarray:
    return np.array(
        [stress_3x3[r, c] for r, c in _VOIGT_FROM_FULL], dtype=np.float64
    )


def _subprocess_pythonpath() -> str:
    root = Path(__file__).resolve().parents[2]
    source = str(root / "python")
    parts = [
        part
        for part in os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if part and part != source
    ]
    return os.pathsep.join(parts)


@pytest.mark.parametrize("species", ("si", "sic"))
def test_tersoff_sic_python_stress_matches_fd(species: str) -> None:
    pos, numbers, box = _silicon_cell() if species == "si" else _sic_cell()
    pot = rgpot.TersoffSiC()
    res = pot.evaluate(pos, numbers, box, compute_stress=True)
    stress = res["stress"]
    assert stress is not None and stress.shape == (3, 3)
    np.testing.assert_allclose(stress, stress.T, atol=1e-13, rtol=0.0)
    fd = _fd_stress_voigt(
        lambda p, b: float(pot.evaluate(p, numbers, b)["energy"]), pos, box
    )
    np.testing.assert_allclose(_full_to_voigt(stress), fd, atol=_FD_TOL, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no TersoffPot in this build")
@pytest.mark.parametrize("species", ("si", "sic"))
def test_tersoff_pot_python_stress_matches_fd(species: str) -> None:
    pos, numbers, box = _silicon_cell() if species == "si" else _sic_cell()
    pot = rgpot.TersoffPot()
    res = pot.evaluate(pos, numbers, box, compute_stress=True)
    stress = res["stress"]
    assert stress is not None and stress.shape == (3, 3)
    np.testing.assert_allclose(stress, stress.T, atol=1e-13, rtol=0.0)
    fd = _fd_stress_voigt(
        lambda p, b: float(pot.evaluate(p, numbers, b)["energy"]), pos, box
    )
    np.testing.assert_allclose(_full_to_voigt(stress), fd, atol=_FD_TOL, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no TersoffPot in this build")
@pytest.mark.parametrize("kind", ("tersoff", "tersoffsic"))
def test_tersoff_ase_calculator_stress_matches_fd(kind: str) -> None:
    pos, numbers, box = _silicon_cell() if kind == "tersoff" else _sic_cell()
    atoms = Atoms(numbers=numbers, positions=pos, cell=box, pbc=True)
    atoms.calc = RgpotCalculator(kind)
    analytic = atoms.get_stress(voigt=True)
    fd = _fd_stress_voigt(
        lambda p, b: float(
            Atoms(
                numbers=numbers,
                positions=p,
                cell=b,
                pbc=True,
                calculator=atoms.calc,
            ).get_potential_energy()
        ),
        pos,
        box,
    )
    np.testing.assert_allclose(analytic, fd, atol=_FD_TOL, rtol=0.0)


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no TersoffPot in this build")
@pytest.mark.parametrize("species", ("si", "sic"))
def test_tersoff_negative_det_and_zero_volume_box(species: str) -> None:
    pos, numbers, box = _silicon_cell() if species == "si" else _sic_cell()
    pot = rgpot.TersoffPot()
    ref = pot.evaluate(pos, numbers, box, compute_stress=True)

    flipped_box = box.copy()
    flipped_box[0] = -flipped_box[0]
    assert np.linalg.det(flipped_box) < 0.0
    lh = pot.evaluate(pos, numbers, flipped_box, compute_stress=True)
    assert lh["energy"] == pytest.approx(ref["energy"], abs=1e-12)
    np.testing.assert_allclose(lh["forces"], ref["forces"], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(lh["stress"], ref["stress"], atol=1e-12, rtol=0.0)

    zero_ortho = np.diag([10.0, 0.0, 10.0])
    if species == "si":
        with pytest.raises(RuntimeError, match="positive volume"):
            pot.evaluate(pos, numbers, zero_ortho, compute_stress=True)
    else:
        zero_res = pot.evaluate(pos, numbers, zero_ortho, compute_stress=True)
        assert zero_res["stress"] is None


@pytest.mark.thread_unsafe
@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no TersoffPot in this build")
@pytest.mark.parametrize("kind", ("tersoff", "tersoffsic"))
def test_tersoff_stress_one_vs_n_threads(kind: str) -> None:
    frames = []
    for seed in (11, 22, 33, 44):
        pos, numbers, box = _silicon_cell(seed) if seed % 2 else _sic_cell(seed)
        frames.append(Atoms(numbers=numbers, positions=pos, cell=box, pbc=True))

    calc = RgpotCalculator(kind)
    rows_1 = calc.calculate_many(
        frames, properties=("energy", "forces", "stress"), threads=1
    )
    rows_4 = calc.calculate_many(
        frames, properties=("energy", "forces", "stress"), threads=4
    )
    assert len(rows_1) == len(rows_4) == len(frames)
    for r1, r4 in zip(rows_1, rows_4):
        assert r1["energy"] == r4["energy"]
        np.testing.assert_array_equal(r1["forces"], r4["forces"])
        np.testing.assert_array_equal(r1["stress"], r4["stress"])

    def eval_one(idx: int):
        p, n, b = _silicon_cell(100 + idx) if idx % 2 else _sic_cell(100 + idx)
        pot = rgpot.TersoffPot() if kind == "tersoff" else rgpot.TersoffSiC()
        return pot.evaluate(p, n, b, compute_stress=True)

    serial = [eval_one(i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(eval_one, range(4)))
    for s, p in zip(serial, parallel):
        assert s["energy"] == p["energy"]
        np.testing.assert_array_equal(s["forces"], p["forces"])
        np.testing.assert_array_equal(s["stress"], p["stress"])


@pytest.mark.thread_unsafe
@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no TersoffPot in this build")
def test_tersoff_stress_omp_one_vs_n_threads() -> None:
    script = (
        "import hashlib, numpy as np, rgpot\n"
        "from ase.build import bulk\n"
        "si = bulk('Si', 'diamond', a=5.432, cubic=True).repeat((2, 2, 2))\n"
        "rng = np.random.default_rng(7)\n"
        "pos = si.positions + rng.uniform(-0.08, 0.08, size=si.positions.shape)\n"
        "box = si.cell.array.copy()\n"
        "box[1, 0] += 0.25\n"
        "box[2, 1] -= 0.15\n"
        "pot = rgpot.TersoffPot()\n"
        "h = hashlib.sha256()\n"
        "for nums in (si.numbers.astype(np.int32), np.where((np.arange(len(si)) % 8) < 4, 6, 14).astype(np.int32)):\n"
        "    res = pot.evaluate(pos, nums, box, compute_stress=True)\n"
        "    h.update(np.asarray([res['energy']], dtype=np.float64).tobytes())\n"
        "    h.update(np.ascontiguousarray(res['forces'], dtype=np.float64).tobytes())\n"
        "    h.update(np.ascontiguousarray(res['stress'], dtype=np.float64).tobytes())\n"
        "print(h.hexdigest())\n"
    )
    digests = []
    for omp in ("1", "4"):
        env = os.environ.copy()
        env["PYTHONPATH"] = _subprocess_pythonpath()
        env["OMP_NUM_THREADS"] = omp
        proc = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        digests.append(proc.stdout.strip().splitlines()[-1])
    assert digests[0] == digests[1]
