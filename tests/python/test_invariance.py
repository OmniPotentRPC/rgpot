"""T5: SHA-256 of raw float64 bytes across Python and OpenMP thread counts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.thread_unsafe

_PYTHON_THREADS = (1, 2, 8)
_OMP_THREADS = (1, 2, 3, 8)


def _interpreters() -> list[str]:
    found = [sys.executable]
    for name in ("python3.14", "python3.14t"):
        path = shutil.which(name)
        if path and path not in found:
            found.append(path)
    return found


def _run(python: str, threads: int, omp: int) -> str:
    worker = Path(__file__).resolve().parent / "invariance_worker.py"
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    src = str(root / "python")
    tests = str(root / "tests" / "python")
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (src, tests, pythonpath) if part
    )
    env["OMP_NUM_THREADS"] = str(omp)
    proc = subprocess.run(
        [python, str(worker), str(threads)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert lines, proc.stdout
    return lines[-1]


def test_rank_invariance_hashes() -> None:
    interpreters = _interpreters()
    reference: str | None = None
    seen: list[tuple[str, int, int, str]] = []
    for python in interpreters:
        for omp in _OMP_THREADS:
            for threads in _PYTHON_THREADS:
                digest = _run(python, threads, omp)
                seen.append((python, threads, omp, digest))
                if reference is None:
                    reference = digest
                assert digest == reference, seen
    assert reference is not None
    print("T5 interpreters", interpreters)
    print("T5 hash", reference)
    print("T5 points", len(seen))
