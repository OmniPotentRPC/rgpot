"""Strong scaling of calculate_many on 64 Pt7 copies with EAM-Al.

Python threads are the BatchEvaluator width. OpenMP threads are a fresh
process per width because OMP_NUM_THREADS is process-global. The Fortran
kernels use ``do concurrent``, so OpenMP is not expected to speed them up.
The numbers are still recorded.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import rgpot

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.skipif(not rgpot.has_fortran_pots, reason="no EAM-Al in this build"),
]

_REPEATS = 5


def _levels(limit: int) -> list[int]:
    chosen: list[int] = []
    for value in (1, 2, 4, max(1, limit)):
        if value not in chosen and value <= max(1, limit):
            chosen.append(value)
        elif value not in chosen:
            chosen.append(max(1, limit))
    ordered: list[int] = []
    for value in chosen:
        if value not in ordered:
            ordered.append(value)
    return ordered


def _summarise(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[0], ordered[-1]


def test_strong_scaling(tmp_path: Path) -> None:
    from rgpot.ase import RgpotCalculator

    from support import pt7, raw_bytes

    frames = [pt7(jitter=0.015, seed=1000 + index) for index in range(64)]
    limit = os.cpu_count() or 1
    widths = _levels(limit)
    calc = RgpotCalculator("eam_al")

    def batch(threads: int):
        started = time.perf_counter()
        rows = calc.calculate_many(frames, properties=("energy", "forces"), threads=threads)
        elapsed = time.perf_counter() - started
        digest = tuple(raw_bytes(row["energy"], row["forces"]) for row in rows)
        return elapsed, digest

    def serial():
        started = time.perf_counter()
        rows = []
        for atoms in frames:
            row = calc.__rg_evaluate__(
                atoms.get_positions(),
                atoms.get_atomic_numbers(),
                atoms.cell.array,
                atoms.pbc,
            )
            rows.append(row)
        elapsed = time.perf_counter() - started
        digest = tuple(raw_bytes(row["energy"], row["forces"]) for row in rows)
        return elapsed, digest

    names = ["serial"] + [f"py-{width}" for width in widths]
    samples: dict[str, list[float]] = {name: [] for name in names}
    digest_ref = None
    for _repeat in range(_REPEATS):
        elapsed, digest = serial()
        samples["serial"].append(elapsed)
        if digest_ref is None:
            digest_ref = digest
        assert digest == digest_ref
        for width in widths:
            elapsed, digest = batch(width)
            samples[f"py-{width}"].append(elapsed)
            assert digest == digest_ref

    omp_samples = _omp_scaling(tmp_path, widths)
    lines = [
        f"interpreter {sys.version.split()[0]}",
        "variant median_s min_s max_s efficiency",
    ]
    serial_median, _serial_min, _serial_max = _summarise(samples["serial"])
    base_median = _summarise(samples["py-1"])[0]
    lines.append(
        "serial(pre-batch N calls) "
        f"{serial_median:.6f} {_serial_min:.6f} {_serial_max:.6f} n/a"
    )
    for width in widths:
        median, low, high = _summarise(samples[f"py-{width}"])
        efficiency = (base_median / median) / width if median > 0 else 0.0
        lines.append(
            f"python-threads-{width} {median:.6f} {low:.6f} {high:.6f} {efficiency:.4f}"
        )
    for width, values in omp_samples.items():
        median, low, high = _summarise(values)
        base = _summarise(omp_samples[1])[0]
        efficiency = (base / median) / width if median > 0 else 0.0
        lines.append(
            f"omp-threads-{width} {median:.6f} {low:.6f} {high:.6f} {efficiency:.4f}"
        )
    py_max = widths[-1]
    py_median = _summarise(samples[f"py-{py_max}"])[0]
    speedup = base_median / py_median if py_median > 0 else 0.0
    if py_max > 1 and speedup > 1.0:
        serial_fraction = (py_max / (py_max - 1.0)) * (1.0 - 1.0 / speedup)
    else:
        serial_fraction = 1.0
    lines.append(
        "limiter 64 copies of Pt7 is a few hundred force components. "
        f"Python-thread speedup at {py_max} is {speedup:.3f} "
        f"(Amdahl serial fraction {serial_fraction:.3f} when speedup exceeds 1). "
        "Thread start, the Python list copy, and the BatchEvaluator join "
        "dominate the EAM kernel. OpenMP does not enter the kernel: the "
        "Fortran ports use do concurrent, not omp parallel."
    )
    text = "\n".join(lines) + "\n"
    (tmp_path / "scaling.txt").write_text(text, encoding="utf-8")
    Path("/tmp/rgpot-b1-scaling.txt").write_text(text, encoding="utf-8")
    print(text)


def _omp_scaling(tmp_path: Path, widths: list[int]) -> dict[int, list[float]]:
    worker = tmp_path / "omp_worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import os, sys, time
            sys.path.insert(0, os.environ["RGPOT_SUPPORT"])
            from support import pt7, raw_bytes
            from rgpot.ase import RgpotCalculator
            frames = [pt7(jitter=0.015, seed=1000 + index) for index in range(64)]
            threads = int(sys.argv[1])
            calc = RgpotCalculator("eam_al")
            calc.calculate_many(frames[:1], threads=1)
            started = time.perf_counter()
            rows = calc.calculate_many(frames, threads=1)
            elapsed = time.perf_counter() - started
            digest = b"".join(raw_bytes(row["energy"], row["forces"]) for row in rows)
            print(f"{elapsed:.9f} {digest.hex()}")
            """
        ).lstrip(),
        encoding="utf-8",
    )
    support = str(Path(__file__).resolve().parent)
    root = Path(__file__).resolve().parents[2]
    src = str(root / "python")
    samples: dict[int, list[float]] = {width: [] for width in widths}
    reference = None
    for _repeat in range(_REPEATS):
        for width in widths:
            env = os.environ.copy()
            pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (src, support, pythonpath) if part
            )
            env["OMP_NUM_THREADS"] = str(width)
            env["RGPOT_SUPPORT"] = support
            proc = subprocess.run(
                [sys.executable, str(worker), str(width)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            assert proc.returncode == 0, proc.stderr
            line = [item for item in proc.stdout.splitlines() if item.strip()][-1]
            elapsed_s, digest = line.split()
            samples[width].append(float(elapsed_s))
            if reference is None:
                reference = digest
            assert digest == reference
    return samples
