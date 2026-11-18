#!/usr/bin/env python3
"""OpenMP crossover sweep for rgpot potentials.

This script builds and compares:
- stock
- team-only threshold patch (kTeamMinAtoms=1, neighbour fallback unchanged)
- fallback-only threshold patch (n<1 fallback, kTeamMinAtoms unchanged)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from typing import Any

TEAM_MIN_ATOMS = 8192
POTENTIAL_CUTOFFS = {
    "lj": 2.5,
    "morse": 6.0,
    "eam": 7.1,
    "emt": 7.1,
    "sw": 3.77118,
    "tersoff": 3.0,
    "zbl": 5.0,
    "sutton-chen": 3.0,
}
POTENTIAL_QUERY_SHAPE = {
    "shell": "half",
    "clustered": False,
}
DEFAULT_POTENTIALS = (
    "lj",
    "morse",
    "eam",
    "emt",
    "sw",
    "tersoff",
    "zbl",
    "sutton-chen",
)


@dataclass(frozen=True)
class BuildContext:
    name: str
    repo: Path
    build_dir: Path
    worker: Path
    sha: str


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def iqr(values: list[float]) -> float:
    return percentile(values, 75.0) - percentile(values, 25.0)


def parse_leading_int(token: str, default: int) -> int:
    match = re.match(r"^\s*([0-9]+)", token)
    if not match:
        return default
    return int(match.group(1))


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def lscpu_info() -> dict[str, int]:
    out = run(["lscpu"], capture_output=True).stdout
    fields: dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    logical = parse_leading_int(fields.get("CPU(s)", "0"), 0)
    tpc = max(1, parse_leading_int(fields.get("Thread(s) per core", "1"), 1))
    cps = parse_leading_int(fields.get("Core(s) per socket", "0"), 0)
    sockets = max(1, parse_leading_int(fields.get("Socket(s)", "1"), 1))
    physical = cps * sockets if cps > 0 else max(1, logical // tpc)
    return {
        "logical_cores": logical,
        "physical_cores": physical,
        "threads_per_core": tpc,
    }


def sibling_key(cpu: int) -> str:
    path = Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return str(cpu)


def exclusive_cpus() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    chosen: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        key = sibling_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(cpu)
    return chosen


def compiler_version(cxx: str) -> str:
    out = run([cxx, "--version"], capture_output=True).stdout.splitlines()
    return out[0].strip() if out else "unknown"


def find_rgpot_library(build_dir: Path) -> Path:
    candidates = sorted(build_dir.rglob("librgpot.so*"))
    if not candidates:
        raise FileNotFoundError(f"librgpot library not found under {build_dir}")
    for cand in candidates:
        if cand.name == "librgpot.so":
            return cand
    return candidates[0]


def configure_and_build(repo: Path, build_dir: Path) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    meson_exe = shutil.which("meson")
    meson_prefix = [meson_exe] if meson_exe else [sys.executable, "-m", "mesonbuild.mesonmain"]
    setup = meson_prefix + [
        "setup",
        str(build_dir),
        str(repo),
        "-Dbuildtype=release",
        "-Dwith_tests=false",
        "-Dwith_examples=false",
        "-Dwith_rpc=false",
        "-Dwith_cache=false",
        "-Dwith_xtensor=false",
        "-Dwith_eigen=false",
        "-Dwith_python=false",
        "-Dwith_fortran_pots=disabled",
        "-Dwith_water=false",
        "-Dwith_ipi=disabled",
        "-Dwith_mpi=disabled",
        "-Dwith_dftd3=false",
        "-Dwith_dftd4=false",
        "-Dwith_expr=false",
        "-Dwith_tblite=false",
        "-Dwith_metatomic=false",
        "-Dwith_xckernel=false",
        "-Dwith_skala=false",
        "-Dwith_long_range=false",
    ]
    env = os.environ.copy()
    env["CXX"] = env.get("CXX", "g++")
    if (build_dir / "meson-private").is_dir():
        run(meson_prefix + ["setup", "--reconfigure", str(build_dir)], env=env)
    else:
        run(setup, env=env)
    run(meson_prefix + ["compile", "-C", str(build_dir)], env=env)


def build_worker(source_repo: Path, link_repo: Path, build_dir: Path, cxx: str) -> Path:
    source = source_repo / "bench" / "rgpot_omp_worker.cc"
    out_dir = build_dir / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    worker = out_dir / "rgpot_omp_worker"
    library = find_rgpot_library(build_dir)
    lib_dir = library.parent
    run(
        [
            cxx,
            "-O3",
            "-DNDEBUG",
            "-std=c++20",
            "-fopenmp",
            str(source),
            "-I" + str(link_repo / "CppCore"),
            "-I" + str(link_repo / "include"),
            "-L" + str(lib_dir),
            "-Wl,-rpath," + str(lib_dir),
            "-lrgpot",
            "-o",
            str(worker),
        ],
        cwd=source_repo,
    )
    return worker


def git_sha(repo: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True).stdout.strip()


def openmp_runtime(worker: Path) -> dict[str, str]:
    out = run(["ldd", str(worker)], capture_output=True).stdout
    runtime_name = "unknown"
    for line in out.splitlines():
        if "libgomp" in line or "libomp" in line:
            lib_match = re.search(r"(libgomp\.so[^\s]*|libomp\.so[^\s]*)", line)
            if lib_match:
                runtime_name = Path(lib_match.group(1)).name
            break
    package_hint = "unknown"
    for pkg in ("libgomp1", "libomp5"):
        try:
            version = run(["dpkg-query", "-W", "-f=${Version}", pkg], capture_output=True).stdout.strip()
        except subprocess.CalledProcessError:
            continue
        if version:
            package_hint = f"{pkg} {version}"
            break
    return {"library": runtime_name, "package_version": package_hint}


def parse_affinity_count(token: str) -> int:
    return len([part for part in token.split(",") if part.strip()]) if token else 0


def run_worker_once(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    sample_seconds: float,
    cpus: list[int],
) -> dict[str, Any]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = ",".join(f"{{{cpu}}}" for cpu in cpus)
    env["GOMP_CPU_AFFINITY"] = ",".join(str(cpu) for cpu in cpus)
    command = [
        "taskset",
        "-c",
        ",".join(str(cpu) for cpu in cpus),
        str(ctx.worker),
        "--potential",
        potential,
        "--atoms",
        str(atoms),
        "--threads",
        str(threads),
        "--samples",
        "1",
        "--sample-seconds",
        f"{sample_seconds:.12g}",
    ]
    out = run(command, cwd=ctx.repo, env=env, capture_output=True)
    return json.loads(out.stdout)


def sample_cell(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    thread_counts: list[int],
    rounds: int,
    sample_seconds: float,
    cpus_by_threads: dict[int, list[int]],
) -> dict[int, dict[str, Any]]:
    rows = {
        t: {
            "build_ns": [],
            "force_ns": [],
            "total_ns": [],
            "pairs_per_atom": [],
            "omp_max": [],
            "omp_team": [],
            "affinity": [],
        }
        for t in thread_counts
    }
    for r in range(rounds):
        order = thread_counts if r % 2 == 0 else list(reversed(thread_counts))
        for t in order:
            row = run_worker_once(
                ctx, potential=potential, atoms=atoms, threads=t, sample_seconds=sample_seconds, cpus=cpus_by_threads[t]
            )
            rows[t]["build_ns"].append(float(row["build_ns_per_eval"][0]))
            rows[t]["force_ns"].append(float(row["force_ns_per_eval"][0]))
            rows[t]["total_ns"].append(float(row["total_ns_per_eval"][0]))
            rows[t]["pairs_per_atom"].append(float(row["nlist_pairs_per_atom_per_eval"][0]))
            rows[t]["omp_max"].append(int(row["omp_max_threads_per_sample"][0]))
            rows[t]["omp_team"].append(int(row["omp_parallel_threads_per_sample"][0]))
            rows[t]["affinity"].append(str(row["affinity_per_sample"][0]))
    return rows


def sample_aa_band(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    rounds: int,
    sample_seconds: float,
    cpus: list[int],
) -> dict[str, tuple[float, float]]:
    force_ratios: list[float] = []
    build_ratios: list[float] = []
    for r in range(rounds):
        order = ("a", "b") if r % 2 == 0 else ("b", "a")
        seen: dict[str, dict[str, float]] = {}
        for arm in order:
            row = run_worker_once(ctx, potential=potential, atoms=atoms, threads=1, sample_seconds=sample_seconds, cpus=cpus)
            seen[arm] = {
                "build": float(row["build_ns_per_eval"][0]),
                "force": float(row["force_ns_per_eval"][0]),
            }
        force_thr_a = 1.0e9 / seen["a"]["force"]
        force_thr_b = 1.0e9 / seen["b"]["force"]
        build_thr_a = 1.0e9 / seen["a"]["build"]
        build_thr_b = 1.0e9 / seen["b"]["build"]
        force_ratios.append(force_thr_b / force_thr_a)
        build_ratios.append(build_thr_b / build_thr_a)
    return {
        "force": (min(force_ratios), max(force_ratios)),
        "build": (min(build_ratios), max(build_ratios)),
    }


def mann_whitney_u(x: list[float], y: list[float]) -> float:
    nx = len(x)
    merged = [(v, 0) for v in x] + [(v, 1) for v in y]
    merged_sorted = sorted(enumerate(merged), key=lambda item: item[1][0])
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged_sorted):
        j = i + 1
        while j < len(merged_sorted) and merged_sorted[j][1][0] == merged_sorted[i][1][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[merged_sorted[k][0]] = avg_rank
        i = j
    rank_sum_x = sum(ranks[i] for i in range(nx))
    return rank_sum_x - nx * (nx + 1) / 2.0


@lru_cache(maxsize=None)
def u_distribution_counts(nx: int, ny: int) -> list[int]:
    n = nx + ny
    max_rank_sum = n * (n + 1) // 2
    dp = [[0] * (max_rank_sum + 1) for _ in range(nx + 1)]
    dp[0][0] = 1
    for rank in range(1, n + 1):
        for k in range(min(nx, rank), 0, -1):
            for s in range(max_rank_sum, rank - 1, -1):
                dp[k][s] += dp[k - 1][s - rank]

    u_counts = [0] * (nx * ny + 1)
    offset = nx * (nx + 1) // 2
    for rank_sum, count in enumerate(dp[nx]):
        if count == 0:
            continue
        u = rank_sum - offset
        if 0 <= u <= nx * ny:
            u_counts[u] += count
    return u_counts


def mann_whitney_one_sided_exact_less(x: list[float], y: list[float]) -> float:
    nx = len(x)
    ny = len(y)
    u = mann_whitney_u(x, y)
    u_floor = int(math.floor(u + 1.0e-12))
    counts = u_distribution_counts(nx, ny)
    total = math.comb(nx + ny, nx)
    tail = sum(counts[: u_floor + 1])
    return tail / float(total)


def format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}g}"


def find_smallest_valid_emt_size(
    ctx: BuildContext,
    candidates: list[int],
    sample_seconds: float,
    cpus: list[int],
) -> int:
    for atoms in sorted(candidates):
        try:
            run_worker_once(ctx, potential="emt", atoms=atoms, threads=1, sample_seconds=sample_seconds, cpus=cpus)
            return atoms
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError("no valid EMT size found in requested range")


def perf_stat(
    ctx: BuildContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    cpus: list[int],
    sample_seconds: float,
) -> dict[str, Any]:
    perf = shutil.which("perf")
    if perf is not None:
        proc = subprocess.run([perf, "--version"], text=True, capture_output=True)
        if proc.returncode != 0 and "perf not found for kernel" in ((proc.stdout or "") + (proc.stderr or "")):
            perf = None
    if perf is None:
        for candidate in (
            "/usr/lib/linux-tools-6.8.0-149/perf",
            "/usr/lib/linux-tools-6.8.0-149-generic/perf",
        ):
            if Path(candidate).is_file():
                perf = candidate
                break
    if perf is None:
        return {"available": False, "reason": "perf not installed"}
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = ",".join(f"{{{cpu}}}" for cpu in cpus)
    env["GOMP_CPU_AFFINITY"] = ",".join(str(cpu) for cpu in cpus)
    cmd = [
        perf,
        "stat",
        "-x,",
        "-e",
        "cache-misses,cache-references,context-switches,cpu-migrations,cycles,instructions",
        "taskset",
        "-c",
        ",".join(str(cpu) for cpu in cpus),
        str(ctx.worker),
        "--potential",
        potential,
        "--atoms",
        str(atoms),
        "--threads",
        str(threads),
        "--samples",
        "1",
        "--sample-seconds",
        f"{sample_seconds:.12g}",
    ]
    proc = subprocess.run(cmd, cwd=str(ctx.repo), env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        text = (proc.stderr or "").strip()
        errno_hint = "unknown"
        if "Operation not permitted" in text:
            errno_hint = "EPERM"
        elif "Permission denied" in text:
            errno_hint = "EACCES"
        return {"available": False, "reason": text or "perf stat failed", "errno_hint": errno_hint}
    metrics: dict[str, float] = {}
    not_supported = False
    for line in proc.stderr.splitlines():
        if "<not supported>" in line:
            not_supported = True
        parts = line.split(",")
        if len(parts) < 3:
            continue
        value, _unit, name = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    if not_supported:
        return {"available": False, "reason": "perf events not supported in this VM", "errno_hint": "ENOTSUP"}
    return {"available": True, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-stock", required=True)
    parser.add_argument("--repo-team", required=True)
    parser.add_argument("--repo-fallback", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--build-stock", required=True)
    parser.add_argument("--build-team", required=True)
    parser.add_argument("--build-fallback", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--sample-seconds", required=True, type=float)
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--potentials", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--team-diff-file", required=True)
    parser.add_argument("--fallback-diff-file", required=True)
    parser.add_argument("--ab-sizes", required=True)
    args = parser.parse_args()

    if args.samples < 12:
        raise RuntimeError("samples must be at least 12")

    repo_stock = Path(args.repo_stock).resolve()
    repo_team = Path(args.repo_team).resolve()
    repo_fallback = Path(args.repo_fallback).resolve()
    build_stock = Path(args.build_stock).resolve()
    build_team = Path(args.build_team).resolve()
    build_fallback = Path(args.build_fallback).resolve()
    out_json = Path(args.out_json).resolve()
    out_md = Path(args.out_md).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    ab_sizes = [int(x) for x in args.ab_sizes.split(",") if x.strip()]
    potentials = [x.strip() for x in args.potentials.split(",") if x.strip()]
    if not potentials:
        potentials = list(DEFAULT_POTENTIALS)

    configure_and_build(repo_stock, build_stock)
    configure_and_build(repo_team, build_team)
    configure_and_build(repo_fallback, build_fallback)
    worker_stock = build_worker(repo_stock, repo_stock, build_stock, args.cxx)
    worker_team = build_worker(repo_stock, repo_team, build_team, args.cxx)
    worker_fallback = build_worker(repo_stock, repo_fallback, build_fallback, args.cxx)
    stock = BuildContext("stock", repo_stock, build_stock, worker_stock, git_sha(repo_stock))
    team = BuildContext("team", repo_team, build_team, worker_team, git_sha(repo_team))
    fallback = BuildContext("fallback", repo_fallback, build_fallback, worker_fallback, git_sha(repo_fallback))

    lscpu = lscpu_info()
    cpus = exclusive_cpus()
    thread_ceiling = min(len(cpus), lscpu["physical_cores"])
    threads = sorted({t for t in (1, 2, 4, 8, thread_ceiling) if t <= thread_ceiling})
    cpus_by_threads = {t: cpus[:t] for t in threads}

    emt_min = find_smallest_valid_emt_size(team, sorted(set(sizes + ab_sizes)), args.sample_seconds, cpus_by_threads[1])

    team_diff = Path(args.team_diff_file).read_text(encoding="utf-8")
    fallback_diff = Path(args.fallback_diff_file).read_text(encoding="utf-8")
    results: list[dict[str, Any]] = []
    aa_bands: list[dict[str, Any]] = []
    crossovers: list[dict[str, Any]] = []
    one_thread_cost: list[dict[str, Any]] = []
    fallback_ab: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    stock_cells: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    team_cells: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}

    for potential in potentials:
        pot_sizes = list(sizes)
        if potential == "emt":
            pot_sizes = sorted({s for s in sizes if s >= emt_min} | {emt_min})
        for atoms in pot_sizes:
            stock_cell = sample_cell(
                stock,
                potential=potential,
                atoms=atoms,
                thread_counts=threads,
                rounds=args.samples,
                sample_seconds=args.sample_seconds,
                cpus_by_threads=cpus_by_threads,
            )
            team_cell = sample_cell(
                team,
                potential=potential,
                atoms=atoms,
                thread_counts=threads,
                rounds=args.samples,
                sample_seconds=args.sample_seconds,
                cpus_by_threads=cpus_by_threads,
            )
            aa = sample_aa_band(
                team,
                potential=potential,
                atoms=atoms,
                rounds=args.samples,
                sample_seconds=args.sample_seconds,
                cpus=cpus_by_threads[1],
            )
            aa_bands.append(
                {
                    "potential": potential,
                    "atoms": atoms,
                    "force_band_low": aa["force"][0],
                    "force_band_high": aa["force"][1],
                    "build_band_low": aa["build"][0],
                    "build_band_high": aa["build"][1],
                }
            )
            stock_cells[(potential, atoms)] = stock_cell
            team_cells[(potential, atoms)] = team_cell
            for t in threads:
                row = {
                    "potential": potential,
                    "atoms": atoms,
                    "threads": t,
                    "stock": {
                        "build_ns_median": statistics.median(stock_cell[t]["build_ns"]),
                        "force_ns_median": statistics.median(stock_cell[t]["force_ns"]),
                        "total_ns_median": statistics.median(stock_cell[t]["total_ns"]),
                        "build_iqr": iqr(stock_cell[t]["build_ns"]),
                        "force_iqr": iqr(stock_cell[t]["force_ns"]),
                        "total_iqr": iqr(stock_cell[t]["total_ns"]),
                    },
                    "team": {
                        "build_ns_median": statistics.median(team_cell[t]["build_ns"]),
                        "force_ns_median": statistics.median(team_cell[t]["force_ns"]),
                        "total_ns_median": statistics.median(team_cell[t]["total_ns"]),
                        "build_iqr": iqr(team_cell[t]["build_ns"]),
                        "force_iqr": iqr(team_cell[t]["force_ns"]),
                        "total_iqr": iqr(team_cell[t]["total_ns"]),
                        "omp_max_observed": sorted(set(team_cell[t]["omp_max"])),
                        "omp_team_observed": sorted(set(team_cell[t]["omp_team"])),
                        "affinity_count_observed": sorted(
                            set(parse_affinity_count(x) for x in team_cell[t]["affinity"])
                        ),
                        "nlist_pairs_per_atom_median": statistics.median(team_cell[t]["pairs_per_atom"]),
                    },
                }
                results.append(row)

            stock_force_1 = 1.0e9 / statistics.median(stock_cell[1]["force_ns"])
            th1_force_1 = 1.0e9 / statistics.median(team_cell[1]["force_ns"])
            stock_build_1 = 1.0e9 / statistics.median(stock_cell[1]["build_ns"])
            th1_build_1 = 1.0e9 / statistics.median(team_cell[1]["build_ns"])
            one_thread_cost.append(
                {
                    "potential": potential,
                    "atoms": atoms,
                    "force_thr_ratio_team_over_stock": th1_force_1 / stock_force_1,
                    "build_thr_ratio_team_over_stock": th1_build_1 / stock_build_1,
                }
            )

        for stage in ("force", "build"):
            stage_key = f"{stage}_ns"
            aa_key = f"{stage}_band_high"
            for target in (2, 8):
                if target not in threads:
                    continue
                crossover: int | None = None
                crossover_speedup = 0.0
                crossover_p = 1.0
                for atoms in pot_sizes:
                    band_row = next(x for x in aa_bands if x["potential"] == potential and x["atoms"] == atoms)
                    band_high = band_row[aa_key]
                    cell = team_cells[(potential, atoms)]
                    ns1 = cell[1][stage_key]
                    nst = cell[target][stage_key]
                    speedup = statistics.median(ns1) / statistics.median(nst)
                    p = mann_whitney_one_sided_exact_less(nst, ns1)
                    min_speedup = 1.2 if target == 2 else 2.0
                    if speedup >= min_speedup and speedup > band_high and p < 0.01:
                        crossover = atoms
                        crossover_speedup = speedup
                        crossover_p = p
                        break
                crossovers.append(
                    {
                        "potential": potential,
                        "stage": stage,
                        "threads": target,
                        "crossover_atoms": crossover,
                        "speedup_at_crossover": crossover_speedup,
                        "p_value": crossover_p,
                    }
                )

        # fallback-only vs stock AB at 1 thread (build time)
        for atoms in ab_sizes:
            if potential == "emt" and atoms < emt_min:
                continue
            stock_build_ns: list[float] = []
            fb_build_ns: list[float] = []
            stock_pairs_per_atom: list[float] = []
            rounds = args.samples
            for r in range(rounds):
                order = ("stock", "fallback") if r % 2 == 0 else ("fallback", "stock")
                for arm in order:
                    ctx = stock if arm == "stock" else fallback
                    row = run_worker_once(
                        ctx, potential=potential, atoms=atoms, threads=1, sample_seconds=args.sample_seconds, cpus=cpus_by_threads[1]
                    )
                    b = float(row["build_ns_per_eval"][0])
                    ppa = float(row["nlist_pairs_per_atom_per_eval"][0])
                    if arm == "stock":
                        stock_build_ns.append(b)
                        stock_pairs_per_atom.append(ppa)
                    else:
                        fb_build_ns.append(b)
            ratio = statistics.median(fb_build_ns) / statistics.median(stock_build_ns)
            p = mann_whitney_one_sided_exact_less(fb_build_ns, stock_build_ns)
            fallback_ab.append(
                {
                    "potential": potential,
                    "atoms": atoms,
                    "build_time_ratio_fallback_over_stock": ratio,
                    "p_fallback_faster_exact": p,
                    "n": len(stock_build_ns),
                    "cutoff": POTENTIAL_CUTOFFS.get(potential, float("nan")),
                    "neighbors_per_atom_stock_median": statistics.median(stock_pairs_per_atom),
                    "shell": POTENTIAL_QUERY_SHAPE["shell"],
                    "clustered": POTENTIAL_QUERY_SHAPE["clustered"],
                }
            )

    # perf counters for requested explanations where available
    diagnostics.append(
        {
            "topic": "sutton-chen-force-2thread",
            "data": {
                "t1": perf_stat(
                    team, potential="sutton-chen", atoms=16384, threads=1, cpus=cpus_by_threads[1], sample_seconds=0.05
                ),
                "t2": perf_stat(
                    team, potential="sutton-chen", atoms=16384, threads=2, cpus=cpus_by_threads[2], sample_seconds=0.05
                )
                if 2 in threads
                else {"available": False, "reason": "thread count not available"},
            },
        }
    )
    diagnostics.append(
        {
            "topic": "lj-force-2thread",
            "data": {
                "t1": perf_stat(
                    team, potential="lj", atoms=16384, threads=1, cpus=cpus_by_threads[1], sample_seconds=0.05
                ),
                "t2": perf_stat(
                    team, potential="lj", atoms=16384, threads=2, cpus=cpus_by_threads[2], sample_seconds=0.05
                )
                if 2 in threads
                else {"available": False, "reason": "thread count not available"},
            },
        }
    )
    diagnostics.append(
        {
            "topic": "tersoff-zbl-build-8thread",
            "data": {
                "tersoff_t1": perf_stat(team, potential="tersoff", atoms=16384, threads=1, cpus=cpus_by_threads[1], sample_seconds=0.05),
                "tersoff_t8": perf_stat(team, potential="tersoff", atoms=16384, threads=8, cpus=cpus_by_threads[8], sample_seconds=0.05) if 8 in threads else {"available": False, "reason": "thread count not available"},
                "zbl_t1": perf_stat(team, potential="zbl", atoms=16384, threads=1, cpus=cpus_by_threads[1], sample_seconds=0.05),
                "zbl_t8": perf_stat(team, potential="zbl", atoms=16384, threads=8, cpus=cpus_by_threads[8], sample_seconds=0.05) if 8 in threads else {"available": False, "reason": "thread count not available"},
            },
        }
    )

    payload = {
        "metadata": {
            "base_sha": args.base_sha,
            "stock_sha": stock.sha,
            "team_sha": team.sha,
            "fallback_sha": fallback.sha,
            "cpu_model": cpu_model(),
            "lscpu": lscpu,
            "thread_counts": threads,
            "sizes_requested": sizes,
            "emt_min_valid_atoms": emt_min,
            "compiler_version": compiler_version(args.cxx),
            "openmp_runtime": openmp_runtime(worker_stock),
            "samples_per_cell": args.samples,
            "sample_seconds": args.sample_seconds,
            "team_patch_diff_text": team_diff,
            "fallback_patch_diff_text": fallback_diff,
            "notes": {
                "emt_force_impl_file_line": "CppCore/rgpot/EMT/EMTPot.cc:39-78",
                "emt_for_each_slice": "No: EMTPot::forceImpl calls EMT methods directly and never enters rgpot::forEachSlice.",
                "nlist_builder_file_line": "CppCore/rgpot/nlist/NeighbourList.cc:656-973",
                "tersoff_zbl_builder": "Both use rgpot::nlist::build with half-shell pair query.",
                "slice_path_1thread_cost_files": "CppCore/rgpot/SuttonChen/SuttonChenPot.cc:177-190, CppCore/rgpot/StillingerWeber/SWSi.cc:267-340, CppCore/rgpot/Tersoff/TersoffSiC.cc:381-470",
                "mw_test_n": args.samples,
            },
            "repro_commands": {
                "neighbour_build_share": "OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential lj --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02",
                "sutton_chen_two_thread_slowdown": "OMP_NUM_THREADS=2 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1}' GOMP_CPU_AFFINITY='0,1' taskset -c 0,1 ./build-crossover-team/bench/rgpot_omp_worker --potential sutton-chen --atoms 16384 --threads 2 --samples 12 --sample-seconds 0.02",
                "emt_force_serial": "OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-team/bench/rgpot_omp_worker --potential emt --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02",
                "tersoff_build_scaling": "OMP_NUM_THREADS=8 OMP_DYNAMIC=false OMP_PROC_BIND=close OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7}' GOMP_CPU_AFFINITY='0,1,2,3,4,5,6,7' taskset -c 0-7 ./build-crossover-stock/bench/rgpot_omp_worker --potential tersoff --atoms 16384 --threads 8 --samples 12 --sample-seconds 0.02",
            },
        },
        "results": results,
        "aa_bands": aa_bands,
        "crossovers": crossovers,
        "one_thread_team_vs_stock": one_thread_cost,
        "fallback_only_ab_vs_stock": fallback_ab,
        "diagnostics": diagnostics,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append("# rgpot OpenMP crossover sweep")
    lines.append("")
    lines.append("## Measurement-only patches (not committed)")
    lines.append("")
    lines.append("### Team-only patch")
    lines.append("")
    lines.append("```diff")
    lines.extend(team_diff.rstrip("\n").splitlines())
    lines.append("```")
    lines.append("")
    lines.append("### Fallback-only patch")
    lines.append("")
    lines.append("```diff")
    lines.extend(fallback_diff.rstrip("\n").splitlines())
    lines.append("```")
    lines.append("")
    lines.append("## Per-potential crossover (team-only build)")
    lines.append("")
    lines.append("|Potential|Stage|Threads|Crossover atoms|Speedup at crossover|p-value|")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in sorted(crossovers, key=lambda r: (r["potential"], r["stage"], r["threads"])):
        lines.append(
            f"|{row['potential']}|{row['stage']}|{row['threads']}|"
            f"{row['crossover_atoms'] if row['crossover_atoms'] is not None else 'none'}|"
            f"{format_float(row['speedup_at_crossover']) if row['crossover_atoms'] is not None else 'n/a'}|"
            f"{format_float(row['p_value'], 8) if row['crossover_atoms'] is not None else 'n/a'}|"
        )
    lines.append("")
    lines.append("## Fallback-only vs stock A/B at 1 thread (build time)")
    lines.append("")
    lines.append("|Potential|Atoms|Cutoff|Shell|Clustered|Neighbors/atom (stock median)|Build-time ratio fallback/stock|Exact one-sided MW p (fallback faster)|n|")
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
    for row in sorted(fallback_ab, key=lambda r: (r["potential"], r["atoms"])):
        lines.append(
            f"|{row['potential']}|{row['atoms']}|{format_float(row['cutoff'])}|{row['shell']}|{str(row['clustered']).lower()}|"
            f"{format_float(row['neighbors_per_atom_stock_median'])}|{format_float(row['build_time_ratio_fallback_over_stock'])}|"
            f"{format_float(row['p_fallback_faster_exact'],8)}|{row['n']}|"
        )
    lines.append("")
    lines.append("## 1-thread throughput cost: team-only vs stock")
    lines.append("")
    lines.append("|Potential|Atoms|Force throughput ratio|Build throughput ratio|")
    lines.append("|---|---:|---:|---:|")
    for row in sorted(one_thread_cost, key=lambda r: (r["potential"], r["atoms"])):
        lines.append(
            f"|{row['potential']}|{row['atoms']}|"
            f"{format_float(row['force_thr_ratio_team_over_stock'])}|"
            f"{format_float(row['build_thr_ratio_team_over_stock'])}|"
        )
    lines.append("")
    lines.append("## Notes requested")
    lines.append("")
    lines.append("- EMT force path does not call `forEachSlice`; it runs through `EMTPot::forceImpl` direct calls (`CppCore/rgpot/EMT/EMTPot.cc:39-78`).")
    lines.append("- Tersoff and ZBL neighbour build rows are from the same internal builder (`rgpot::nlist::build`, `CppCore/rgpot/nlist/NeighbourList.cc:656-973`) and the same query shape (`half`, non-clustered); the main measured difference across potentials is per-atom neighbour count from cutoff and lattice.")
    lines.append("- One-sided MW p-values are exact under the null for n=12 vs n=12.")
    lines.append("")
    lines.append("## Reproduction commands")
    lines.append("")
    repro = payload["metadata"]["repro_commands"]
    for key in ("neighbour_build_share", "sutton_chen_two_thread_slowdown", "emt_force_serial", "tersoff_build_scaling"):
        lines.append(f"- `{key}`: `{repro[key]}`")
    lines.append("")
    lines.append("## Optional perf counters")
    lines.append("")
    for diag in diagnostics:
        lines.append(f"- {diag['topic']}: `{json.dumps(diag['data'], sort_keys=True)}`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
