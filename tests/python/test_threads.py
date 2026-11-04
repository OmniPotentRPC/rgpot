"""T1: concurrent callers, the F3 claim, and the reentrant protocol."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import rgpot
from rgpot.ase import RgpotCalculator
from rgpot.ase.claim import ConnectionClaim

from support import lj_dimer, pt7, raw_bytes


pytestmark = pytest.mark.thread_unsafe


def _eval(calc: RgpotCalculator, atoms) -> bytes:
    row = calc.__rg_evaluate__(
        atoms.get_positions(),
        atoms.get_atomic_numbers(),
        np.asarray(atoms.cell.array),
        atoms.pbc,
    )
    return raw_bytes(row["energy"], row["forces"])


def test_independent_calculators() -> None:
    atoms = lj_dimer()
    reference = _eval(RgpotCalculator("lj"), atoms)

    def work() -> bytes:
        folder = Path(tempfile.mkdtemp())
        calc = RgpotCalculator("lj", directory=str(folder))
        return _eval(calc, atoms)

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _index: work(), range(8)))
    assert rows == [reference] * 8


def test_shared_lj_object() -> None:
    pot = rgpot.LJPot(u0=1.1, cutoff=8.0, psi=0.8)
    calc = RgpotCalculator(pot)
    atoms = lj_dimer(1.7)
    reference = _eval(calc, atoms)
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _index: _eval(calc, atoms), range(8)))
    assert rows == [reference] * 8


@pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build")
def test_one_eam_object_from_many_threads() -> None:
    calc = RgpotCalculator("eam_al")
    frames = [pt7(jitter=0.01, seed=index) for index in range(8)]
    serial = [_eval(calc, atoms) for atoms in frames]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(lambda atoms: _eval(calc, atoms), frames))
    assert parallel == serial


def test_connection_claim_raises() -> None:
    claim = ConnectionClaim("RgpotRpcCalculator")
    start = threading.Barrier(8)
    release = threading.Event()
    errors: list[str] = []
    entered: list[int] = []

    def worker() -> None:
        start.wait()
        try:
            with claim:
                entered.append(threading.get_ident())
                release.wait(timeout=5)
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _index in range(8)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (len(entered) < 1 or len(errors) < 7):
        time.sleep(0.01)
    release.set()
    for thread in threads:
        thread.join()
    assert len(entered) == 1
    assert len(errors) == 7
    assert all("is in use by another thread" in text for text in errors)


def test_reentrant_protocol_matches_serial() -> None:
    calc = RgpotCalculator("lj")
    assert calc.protocol_reentrant is True
    frames = [lj_dimer(distance) for distance in (1.1, 1.4, 1.9, 2.4)]

    def run(threads: int, protocol_reentrant: bool) -> list[bytes]:
        calc.protocol_reentrant = protocol_reentrant
        if threads == 1 or not calc.protocol_reentrant:
            return [_eval(calc, atoms) for atoms in frames]
        with ThreadPoolExecutor(max_workers=threads) as pool:
            return list(pool.map(lambda atoms: _eval(calc, atoms), frames))

    serial = run(1, True)
    assert run(8, True) == serial
    assert run(8, False) == serial
    calc.protocol_reentrant = True
