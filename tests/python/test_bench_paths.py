"""Each benchmark row reaches the kernel it claims to time.

A cluster that fits inside the cutoff takes the serial direct walk, so
the thread count never applies. A threaded kernel binds more than one
core. A second call on the same geometry must still follow a moved atom.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

# The suite tree's python/ package has no extension module. Drop it so
# the installed wheel is the library these rows call.
_ROOT = Path(__file__).resolve().parents[2]
sys.path[:] = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() not in {_ROOT, _ROOT / "python"}
]
import rgpot

_SPEC = importlib.util.spec_from_file_location(
    "bench_threaded", _ROOT / "benchmarks" / "threaded.py"
)
_BENCH = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_BENCH)

_CLUSTER_CUTOFF = 15.0


def _diagonal(positions: np.ndarray) -> float:
    span = positions.max(axis=0) - positions.min(axis=0)
    return float(math.sqrt(float((span * span).sum())))


def _child_cluster(atoms: int) -> np.ndarray:
    start = _BENCH._CHILD.index("def _stretch_past_cutoff")
    stop = _BENCH._CHILD.index("def build(")
    namespace = {"np": np}
    exec(_BENCH._CHILD[start:stop], namespace)
    positions, _types, _box = namespace["lj_cluster"](atoms)
    return positions


def _geometry(kind: str, atoms: int):
    if kind == "LJCluster":
        return _BENCH._lj_cluster(atoms), rgpot.LJClusterPot(cutoff=_CLUSTER_CUTOFF)
    if kind == "LJ":
        spacing = 2.0 ** (1.0 / 6.0) * math.sqrt(2.0)
        return _BENCH._lattice(atoms, spacing, _BENCH._FCC, 1), rgpot.LJPot(cutoff=2.0)
    if kind == "Morse":
        return _BENCH._lattice(atoms, 4.05, _BENCH._FCC, 1), rgpot.MorsePot(cutoff=6.0)
    if kind == "SW":
        return _BENCH._diamond(atoms), rgpot.StillingerWeberSi()
    if kind == "Tersoff":
        return _BENCH._diamond(atoms), rgpot.TersoffSiC()
    return _BENCH._aluminium(atoms), rgpot.EAMAlPot()


def _energy(pot, positions, types, box) -> float:
    energy, forces, _variance = pot(positions, types, box)
    assert forces.shape == positions.shape
    assert math.isfinite(float(energy))
    return float(energy)


def _parallel(kind: str, atoms: int, threads: int) -> bool:
    """True when this row's kernel opens a team of more than one thread."""
    if threads <= 1:
        return False
    if kind in ("LJCluster", "SW", "Tersoff"):
        return True
    # Morse stays on one thread at 256 atoms and below four threads.
    if kind == "Morse":
        return atoms > 256 and threads >= 4
    return False


def _bound_cpus(kind: str, atoms: int, threads: int) -> set[str]:
    cores = _BENCH._worker_cores(threads)
    script = textwrap.dedent(
        """
        import importlib.util, os, time
        from pathlib import Path
        root = Path(os.environ["BENCH_ROOT"])
        spec = importlib.util.spec_from_file_location(
            "bench_threaded", root / "benchmarks" / "threaded.py"
        )
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        import rgpot
        kind = os.environ["BENCH_KIND"]
        atoms = int(os.environ["BENCH_ATOMS"])
        if kind == "LJCluster":
            pos, types, box = bench._lj_cluster(atoms)
            pot = rgpot.LJClusterPot(cutoff=15.0)
        elif kind == "Morse":
            pos, types, box = bench._lattice(atoms, 4.05, bench._FCC, 1)
            pot = rgpot.MorsePot(cutoff=6.0)
        elif kind == "SW":
            pos, types, box = bench._diamond(atoms)
            pot = rgpot.StillingerWeberSi()
        else:
            pos, types, box = bench._diamond(atoms)
            pot = rgpot.TersoffSiC()
        for _ in range(2):
            pot(pos, types, box)
        seen = set()
        stop = {"v": False}

        def masks():
            found = []
            for name in os.listdir("/proc/self/task"):
                try:
                    text = open(
                        "/proc/self/task/%s/status" % name,
                        encoding="ascii",
                        errors="replace",
                    ).read()
                except OSError:
                    continue
                for line in text.splitlines():
                    if line.startswith("Cpus_allowed_list:"):
                        found.append(line.split()[-1])
            return tuple(sorted(found))

        import threading
        def watch():
            while not stop["v"]:
                seen.add(masks())
        thread = threading.Thread(target=watch)
        thread.start()
        started = time.perf_counter()
        while time.perf_counter() - started < 0.25:
            pot(pos, types, box)
        stop["v"] = True
        thread.join()
        cpus = set()
        for snap in seen:
            for token in snap:
                if token.isdigit():
                    cpus.add(token)
        print(" ".join(sorted(cpus)))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "BENCH_ROOT": str(_ROOT),
            "BENCH_KIND": kind,
            "BENCH_ATOMS": str(atoms),
            "OMP_NUM_THREADS": str(threads),
            "OMP_DYNAMIC": "false",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": ",".join("{%d}" % cpu for cpu in cores),
        }
    )
    completed = subprocess.run(
        [
            "taskset",
            "-c",
            ",".join(str(cpu) for cpu in cores),
            sys.executable,
            "-c",
            script,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    line = completed.stdout.strip().splitlines()[-1]
    return set(line.split()) if line else set()


@pytest.mark.parametrize("atoms", _BENCH.ATOMS)
def test_cluster_does_not_fit_inside_the_cutoff(atoms: int) -> None:
    positions, _types, _box = _BENCH._lj_cluster(atoms)
    assert _diagonal(positions) > _CLUSTER_CUTOFF
    child = _child_cluster(atoms)
    assert _diagonal(child) > _CLUSTER_CUTOFF


@pytest.mark.parametrize("kind", ["LJCluster", "LJ", "Morse", "SW", "Tersoff", "EAM"])
@pytest.mark.parametrize("atoms", _BENCH.ATOMS)
def test_repeat_call_follows_a_moved_atom(kind: str, atoms: int) -> None:
    if kind == "EAM" and rgpot.EAMAlPot is None:
        pytest.skip("this build has no stored EAM potential")
    (positions, types, box), pot = _geometry(kind, atoms)
    first = _energy(pot, positions, types, box)
    second = _energy(pot, positions, types, box)
    assert first == second
    assert first != 0.0
    moved = positions.copy()
    moved[0, 0] += 0.2
    assert _energy(pot, moved, types, box) != first


@pytest.mark.parametrize("kind", ["LJCluster", "Morse", "SW", "Tersoff"])
@pytest.mark.parametrize("atoms", _BENCH.ATOMS)
@pytest.mark.parametrize("threads", _BENCH.THREADS)
def test_threaded_row_binds_a_team(kind: str, atoms: int, threads: int) -> None:
    if not _parallel(kind, atoms, threads):
        return
    cpus = _bound_cpus(kind, atoms, threads)
    assert len(cpus) >= 2


def test_parent_pin_does_not_shrink_the_next_team() -> None:
    bench = _BENCH.LJCluster()
    bench.setup(256, 1)
    bench.setup(256, 4)
    assert bench.env["BENCH_CORES"] == "4,5,6,7"


def test_short_call_gets_a_long_sample() -> None:
    # A 50 us call needs thousands of repetitions before a 4 ms wait is
    # half a percent of the sample. A call that is already long stays
    # at the floor.
    assert _BENCH._sample_calls(50e-6, 256) >= 10000
    assert _BENCH._sample_calls(0.2, 4096) == _BENCH._loops(4096)


# Two timed windows of a settled row stay inside this band. The cold
# first window is about twice the next one, which is outside it.
_AA_LOW = 0.85
_AA_HIGH = 1.20


def _window_ratio(kind: str, atoms: int, threads: int) -> float:
    env = dict(_BENCH._team_env(threads))
    seconds = _BENCH._steady_seconds(kind, atoms, threads, env)
    calls = _BENCH._sample_calls(seconds, atoms)
    env["BENCH_CALLS"] = str(calls)
    text = _BENCH._run_child(
        kind,
        atoms,
        threads,
        env,
        f"""
def _window():
    started = time.perf_counter()
    for _ in range({calls}):
        call(pot, positions, atom_types, box)
    return (time.perf_counter() - started) / {calls}


first = _window()
second = _window()
print(second / first)
""",
    )
    return float(text.splitlines()[-1])


@pytest.mark.parametrize("kind", ["LJCluster", "LJ", "Morse", "SW", "Tersoff", "EAM"])
@pytest.mark.parametrize("atoms", _BENCH.ATOMS)
@pytest.mark.parametrize("threads", _BENCH.THREADS)
def test_timed_windows_agree_within_the_aa_band(
    kind: str, atoms: int, threads: int
) -> None:
    if kind == "EAM" and rgpot.EAMAlPot is None:
        pytest.skip("this build has no stored EAM potential")
    # The harness kind for Stillinger-Weber is the class name, not the
    # short label the geometry helper uses.
    kernel = "StillingerWeber" if kind == "SW" else kind
    ratio = _window_ratio(kernel, atoms, threads)
    assert _AA_LOW <= ratio <= _AA_HIGH
