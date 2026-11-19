#!/usr/bin/env python3
"""Resumable pre-landing performance harness for rgpot PR #287 and PR #293.

Protocol:
- Builds three independent release arms per PR: control, control_aa (A/A), head
- Records git commit SHA and tree SHA for every arm
- Audits compiler flags (no -march=native, no -ffast-math)
- Pins 1-thread runs with taskset to one core and every thread env var set to 1
- Pins multi-thread runs to exact core count with OMP_PROC_BIND=close
- Randomizes arm order per round from shuffled permutation blocks and records (round, slot, arm)
- Uses a 14-round pilot (no warm-up discards, no --quick, no rescue runs)
- Computes adaptive n (cap 48) for relative SE <= 3% on head/control ratio
- Reports per-cell median, IQR, ratio with 95% bootstrap CI, symmetrized A/A band,
  exact two-sided Mann-Whitney p, Bonferroni-adjusted p, and regression/win/neutral verdict
- Verifies small-cell correctness (#287) and force agreement + determinism (#293)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DEVELOP_SHA = "4aa93511d35c1cc75a93b0b372eb80c55f1365d8"

PR_SPECS: dict[str, dict[str, Any]] = {
    "pr287": {
        "pr_number": 287,
        "title": "fix(nlist): multi-image neighbour lists for small cells (r5b)",
        "control_sha": "8b55e633109da481951bd0e4343918d37cee7fd0",
        "head_sha": "1547b9091cbf42c1aafc52aa0a42c35be5629051",
    },
    "pr293": {
        "pr_number": 293,
        "title": "fix(sutton-chen): parallel forces under a finite term limit (r5)",
        "control_sha": "4aa93511d35c1cc75a93b0b372eb80c55f1365d8",
        "head_sha": "a307ad422e460ce3956032d725917692f9b02af3",
    },
}

ARMS = ("control", "control_aa", "head")
PILOT_ROUNDS = 14
MAX_ROUNDS = 48
TARGET_REL_SE = 0.03
BOOTSTRAP_PILOT = 2000
BOOTSTRAP_FINAL = 10000
BONFERRONI_ALPHA = 0.01

SINGLE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TOKIO_WORKER_THREADS",
)

RAW_CSV_FIELDS = (
    "pr",
    "cell_id",
    "cell_kind",
    "potential",
    "atoms",
    "threads",
    "round_idx",
    "slot_idx",
    "arm",
    "commit_sha",
    "tree_sha",
    "total_ns_per_eval",
    "evals",
    "omp_max_threads",
    "omp_parallel_threads",
    "affinity",
    "omp_parallel_affinity",
    "energy",
    "energy_bits",
    "force_hash",
    "list_reused",
    "repeat_energy_bit_identical",
    "repeat_force_bit_identical",
    "repeat_max_abs_force_diff",
    "small_cell_regime",
)

_UNTIED_MW_CACHE: dict[tuple[int, int], list[int]] = {}


@dataclass(frozen=True)
class CellSpec:
    pr: str
    cell_id: str
    cell_kind: str  # "comparative" or "small_head_only"
    potential: str
    atoms: int
    threads: int
    cutoff: float | None = None
    term_limit: int | None = None
    verify_small_cell: bool = False


@dataclass(frozen=True)
class ArmContext:
    pr: str
    arm: str
    worktree: Path
    build_dir: Path
    worker: Path
    commit_sha: str
    tree_sha: str


def pr287_cells() -> list[CellSpec]:
    cells: list[CellSpec] = []
    for potential in ("nlist_build", "eam", "morse", "lj"):
        for atoms in (32, 256, 1024, 4096):
            cells.append(
                CellSpec(
                    pr="pr287",
                    cell_id=f"{potential}_n{atoms}_t1",
                    cell_kind="comparative",
                    potential=potential,
                    atoms=atoms,
                    threads=1,
                    verify_small_cell=(atoms == 32),
                )
            )
    for atoms in (108, 256, 1024, 4096):
        cells.append(
            CellSpec(
                pr="pr287",
                cell_id=f"eam_table_n{atoms}_t1",
                cell_kind="comparative",
                potential="eam_table",
                atoms=atoms,
                threads=1,
                verify_small_cell=False,
            )
        )
    for atoms in (4, 8, 16, 32):
        cells.append(
            CellSpec(
                pr="pr287",
                cell_id=f"eam_table_small_n{atoms}_t1",
                cell_kind="small_head_only",
                potential="eam_table",
                atoms=atoms,
                threads=1,
                verify_small_cell=True,
            )
        )
    cells.append(
        CellSpec(
            pr="pr287",
            cell_id="eam_table_tri3_small_n3_t1",
            cell_kind="small_head_only",
            potential="eam_table_tri3",
            atoms=3,
            threads=1,
            verify_small_cell=True,
        )
    )
    return cells


def pr293_cells() -> list[CellSpec]:
    cells: list[CellSpec] = []
    for atoms in (256, 1024, 4096, 10976):
        for threads in (1, 2, 4, 8):
            cells.append(
                CellSpec(
                    pr="pr293",
                    cell_id=f"sutton_chen_n{atoms}_t{threads}",
                    cell_kind="comparative",
                    potential="sutton-chen",
                    atoms=atoms,
                    threads=threads,
                )
            )
    return cells


def all_cells() -> list[CellSpec]:
    return pr287_cells() + pr293_cells()


def run_cmd(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
    preexec_fn: Any = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
        text=True,
        capture_output=capture_output,
        preexec_fn=preexec_fn,
    )


def sibling_key(cpu: int) -> str:
    path = Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology" / "thread_siblings_list"
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        return str(cpu)


def exclusive_cpus() -> list[int]:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = list(range(os.cpu_count() or 1))
    chosen: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        key = sibling_key(cpu)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(cpu)
    return chosen


def build_pinning_env(
    threads: int, cpus: list[int], base_env: dict[str, str] | None = None
) -> tuple[list[int], dict[str, str]]:
    if threads < 1:
        raise ValueError("threads must be >= 1")
    if len(cpus) < threads:
        raise RuntimeError(f"requested {threads} cores, only {len(cpus)} available")
    chosen = list(cpus[:threads])
    env = dict(base_env if base_env is not None else os.environ)
    cpu_list = ",".join(str(c) for c in chosen)
    if threads == 1:
        for var in SINGLE_THREAD_ENV_VARS:
            env[var] = "1"
    else:
        env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "false"
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = ",".join(f"{{{c}}}" for c in chosen)
    env["GOMP_CPU_AFFINITY"] = cpu_list
    return chosen, env


def make_round_schedule(
    n_rounds: int, seed: int, arms: tuple[str, ...] = ARMS
) -> list[tuple[str, ...]]:
    rng = random.Random(seed)
    perms = list(itertools.permutations(arms))
    schedule: list[tuple[str, ...]] = []
    while len(schedule) < n_rounds:
        block = list(perms)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:n_rounds]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
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


def bootstrap_ratio_ci(
    num_samples: list[float],
    den_samples: list[float],
    *,
    resamples: int = BOOTSTRAP_FINAL,
    seed: int = 287293,
) -> dict[str, float]:
    if not num_samples or not den_samples:
        raise ValueError("bootstrap requires non-empty samples")
    n_num = len(num_samples)
    n_den = len(den_samples)
    ratio = statistics.median(num_samples) / statistics.median(den_samples)
    rng = random.Random(seed)
    boot_ratios: list[float] = []
    for _ in range(resamples):
        s_num = [num_samples[rng.randrange(n_num)] for _ in range(n_num)]
        s_den = [den_samples[rng.randrange(n_den)] for _ in range(n_den)]
        boot_ratios.append(statistics.median(s_num) / statistics.median(s_den))
    ci_low = percentile(boot_ratios, 2.5)
    ci_high = percentile(boot_ratios, 97.5)
    mean_b = sum(boot_ratios) / float(len(boot_ratios))
    var_b = sum((x - mean_b) ** 2 for x in boot_ratios) / float(max(1, len(boot_ratios) - 1))
    se = math.sqrt(var_b)
    rel_se = se / ratio if ratio > 0.0 else float("inf")
    return {
        "ratio": ratio,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "se": se,
        "rel_se": rel_se,
    }


def bootstrap_single_rel_se(
    samples: list[float], *, resamples: int = BOOTSTRAP_PILOT, seed: int = 287293
) -> float:
    n = len(samples)
    med = statistics.median(samples)
    rng = random.Random(seed)
    boot_meds = [
        statistics.median([samples[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    ]
    mean_b = sum(boot_meds) / float(len(boot_meds))
    var_b = sum((x - mean_b) ** 2 for x in boot_meds) / float(max(1, len(boot_meds) - 1))
    return math.sqrt(var_b) / med if med > 0.0 else float("inf")


def compute_n_needed(
    pilot_rel_se: float,
    *,
    pilot_rounds: int = PILOT_ROUNDS,
    target_rel_se: float = TARGET_REL_SE,
    cap: int = MAX_ROUNDS,
) -> int:
    if not math.isfinite(pilot_rel_se):
        return cap
    if pilot_rel_se <= target_rel_se:
        return pilot_rounds
    raw = math.ceil(pilot_rounds * (pilot_rel_se / target_rel_se) ** 2)
    return min(cap, max(pilot_rounds, int(raw)))


def symmetrized_aa_band(
    aa_samples: list[float],
    ctrl_samples: list[float],
    *,
    resamples: int = BOOTSTRAP_FINAL,
    seed: int = 287294,
) -> dict[str, float]:
    stats = bootstrap_ratio_ci(aa_samples, ctrl_samples, resamples=resamples, seed=seed)
    lo = stats["ci_low"]
    hi = stats["ci_high"]
    band_low = min(lo, 1.0 / hi if hi > 0.0 else lo, 1.0)
    band_high = max(hi, 1.0 / lo if lo > 0.0 else hi, 1.0)
    per_round = [a / c for a, c in zip(aa_samples, ctrl_samples) if c > 0.0]
    return {
        "aa_ratio": stats["ratio"],
        "aa_ci_low": lo,
        "aa_ci_high": hi,
        "band_low": band_low,
        "band_high": band_high,
        "per_round_min": min(per_round) if per_round else band_low,
        "per_round_max": max(per_round) if per_round else band_high,
    }


def _untied_mw_distribution(nx: int, ny: int) -> list[int]:
    if nx > ny:
        nx, ny = ny, nx
    key = (nx, ny)
    cached = _UNTIED_MW_CACHE.get(key)
    if cached is not None:
        return cached
    # Partition recurrence for q-binomial [nx + ny choose nx]_q.
    # dp[k][u] = number of subsets of size k from {1..i} with inversion count u.
    u_max = nx * ny
    dp = [[0] * (u_max + 1) for _ in range(nx + 1)]
    dp[0][0] = 1
    total = nx + ny
    for i in range(1, total + 1):
        max_k = min(i, nx)
        for k in range(max_k, 0, -1):
            y_before = i - k
            if y_before > ny:
                continue
            prev_row = dp[k - 1]
            cur_row = dp[k]
            limit_u = min(u_max - y_before, (k - 1) * ny)
            for u in range(limit_u + 1):
                cnt = prev_row[u]
                if cnt:
                    cur_row[u + y_before] += cnt
    _UNTIED_MW_CACHE[key] = dp[nx]
    return dp[nx]


def exact_mann_whitney_two_sided(x: list[float], y: list[float]) -> dict[str, float]:
    nx = len(x)
    ny = len(y)
    if nx == 0 or ny == 0:
        return {"u": float("nan"), "p_two_sided": float("nan")}

    merged = sorted([(v, 0) for v in x] + [(v, 1) for v in y], key=lambda item: item[0])
    groups: list[tuple[int, int]] = []
    i = 0
    total_n = nx + ny
    rank_sum_x2 = 0  # doubled rank sum for x
    while i < total_n:
        j = i + 1
        while j < total_n and merged[j][0] == merged[i][0]:
            j += 1
        t = j - i
        gx = sum(1 for k in range(i, j) if merged[k][1] == 0)
        # doubled average rank = (i + 1) + j
        rank_sum_x2 += gx * (i + 1 + j)
        groups.append((t, gx))
        i = j

    u2_obs = rank_sum_x2 - nx * (nx + 1)
    u1 = u2_obs / 2.0
    total_ways = math.comb(total_n, nx)

    has_ties = any(t > 1 for t, _ in groups)
    if not has_ties:
        dist = _untied_mw_distribution(nx, ny)
        u_obs = u2_obs // 2
        u_max = nx * ny
        u_sym = u_max - u_obs
        if u_obs == u_sym:
            tail = sum(dist)
        else:
            lo = min(u_obs, u_sym)
            hi = max(u_obs, u_sym)
            tail = sum(dist[: lo + 1]) + sum(dist[hi:])
        p_val = min(1.0, float(tail) / float(total_ways))
        return {"u": u1, "p_two_sided": p_val}

    # Exact DP over tied groups in doubled-U coordinates (always integer).
    u2_max = 2 * nx * ny
    u2_sym = u2_max - u2_obs
    dp: list[dict[int, int]] = [dict() for _ in range(nx + 1)]
    dp[0][0] = 1
    seen = 0
    for t, _ in groups:
        new_dp: list[dict[int, int]] = [dict() for _ in range(nx + 1)]
        for k_prev in range(min(seen, nx) + 1):
            row = dp[k_prev]
            if not row:
                continue
            y_prev = seen - k_prev
            max_a = min(t, nx - k_prev)
            for a in range(max_a + 1):
                if (seen + t) - (k_prev + a) > ny:
                    continue
                mult = math.comb(t, a)
                du2 = a * (2 * y_prev + t - a)
                dst = new_dp[k_prev + a]
                for u2, ways in row.items():
                    key = u2 + du2
                    dst[key] = dst.get(key, 0) + ways * mult
        dp = new_dp
        seen += t

    dist_tied = dp[nx]
    if u2_obs == u2_sym:
        p_val = 1.0
    else:
        lo = min(u2_obs, u2_sym)
        hi = max(u2_obs, u2_sym)
        tail = sum(ways for u2, ways in dist_tied.items() if u2 <= lo or u2 >= hi)
        p_val = min(1.0, float(tail) / float(total_ways))
    return {"u": u1, "p_two_sided": p_val}


def classify_cell_verdict(
    *,
    ratio_ci_low: float,
    ratio_ci_high: float,
    aa_band_low: float,
    aa_band_high: float,
    bonferroni_p: float,
    alpha: float = BONFERRONI_ALPHA,
) -> str:
    if bonferroni_p < alpha:
        if ratio_ci_low > aa_band_high:
            return "regression"
        if ratio_ci_high < aa_band_low:
            return "win"
    return "neutral"


def aggregate_pr_verdict(cell_verdicts: list[str]) -> str:
    if any(v == "regression" for v in cell_verdicts):
        return "regression"
    if any(v == "win" for v in cell_verdicts):
        return "win"
    return "neutral"


def git_rev_parse(repo: Path, ref: str) -> str:
    return run_cmd(["git", "rev-parse", ref], cwd=repo).stdout.strip()


def prepare_worktree(main_repo: Path, wt_path: Path, expected_sha: str) -> tuple[str, str]:
    if wt_path.exists():
        got = git_rev_parse(wt_path, "HEAD")
        if got != expected_sha:
            raise RuntimeError(f"worktree {wt_path} at {got}, expected {expected_sha}")
    else:
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            ["git", "worktree", "add", "--detach", str(wt_path), expected_sha],
            cwd=main_repo,
        )
    commit_sha = git_rev_parse(wt_path, "HEAD")
    tree_sha = git_rev_parse(wt_path, "HEAD^{tree}")
    if commit_sha != expected_sha:
        raise RuntimeError(f"commit mismatch in {wt_path}: {commit_sha} != {expected_sha}")
    return commit_sha, tree_sha


def audit_build_flags(build_dir: Path) -> dict[str, Any]:
    cc_path = build_dir / "compile_commands.json"
    data = json.loads(cc_path.read_text(encoding="utf-8"))
    has_native = False
    has_fast_math = False
    has_o3 = False
    openmp_files = 0
    for entry in data:
        cmd = " ".join(entry.get("arguments", [])) or str(entry.get("command", ""))
        if "-march=native" in cmd or "-mcpu=native" in cmd:
            has_native = True
        if "-ffast-math" in cmd and "-fno-fast-math" not in cmd:
            has_fast_math = True
        if "-O3" in cmd:
            has_o3 = True
        if "-fopenmp" in cmd:
            openmp_files += 1
    if has_native or has_fast_math:
        raise RuntimeError(
            f"forbidden compiler flag detected in {build_dir}: "
            f"native={has_native}, fast_math={has_fast_math}"
        )
    return {
        "compile_units": len(data),
        "has_O3": has_o3,
        "has_march_native": has_native,
        "has_ffast_math": has_fast_math,
        "openmp_units": openmp_files,
    }


def find_librgpot(build_dir: Path) -> Path:
    for name in ("librgpot.so", "librgpot.dylib"):
        matches = sorted(build_dir.rglob(name))
        if matches:
            return matches[0]
    matches = sorted(build_dir.rglob("librgpot.so*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"librgpot not found under {build_dir}")


def build_arm(
    *,
    main_repo: Path,
    pr: str,
    arm: str,
    expected_sha: str,
    worktree_root: Path,
    build_root: Path,
    worker_source: Path,
    cc: str,
    cxx: str,
) -> ArmContext:
    wt_path = (worktree_root / f"{pr}_{arm}").resolve()
    build_dir = (build_root / f"{pr}_{arm}").resolve()
    commit_sha, tree_sha = prepare_worktree(main_repo, wt_path, expected_sha)

    build_dir.mkdir(parents=True, exist_ok=True)
    meson_exe = shutil.which("meson")
    if meson_exe is None:
        raise RuntimeError("meson executable not found on PATH")

    env = os.environ.copy()
    env["CC"] = cc
    env["CXX"] = cxx

    if not (build_dir / "meson-private").is_dir():
        setup_cmd = [
            meson_exe,
            "setup",
            str(build_dir),
            str(wt_path),
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
            "-Dwith_eam_table=true",
            "-Dwith_fast_nlist=enabled",
        ]
        run_cmd(setup_cmd, env=env)
    run_cmd([meson_exe, "compile", "-C", str(build_dir)], env=env)

    audit_build_flags(build_dir)
    lib_path = find_librgpot(build_dir)
    lib_dir = lib_path.parent
    worker_dir = build_dir / "bench"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_bin = worker_dir / "rgpot_perf_worker"

    compile_worker = [
        cxx,
        "-O3",
        "-DNDEBUG",
        "-std=c++20",
        "-fopenmp",
        "-DRGPOT_HAS_EAM_TABLE",
        str(worker_source),
        "-I" + str(wt_path / "CppCore"),
        "-I" + str(wt_path / "include"),
        "-L" + str(lib_dir),
        "-Wl,-rpath," + str(lib_dir),
        "-lrgpot",
        "-o",
        str(worker_bin),
    ]
    run_cmd(compile_worker, cwd=wt_path, env=env)

    ldd_out = run_cmd(["ldd", str(worker_bin)]).stdout
    if str(lib_dir) not in ldd_out:
        raise RuntimeError(f"worker {worker_bin} did not link librgpot from {lib_dir}")

    return ArmContext(
        pr=pr,
        arm=arm,
        worktree=wt_path,
        build_dir=build_dir,
        worker=worker_bin,
        commit_sha=commit_sha,
        tree_sha=tree_sha,
    )


def run_worker_once(
    ctx: ArmContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    sample_seconds: float,
    cpus: list[int],
    cutoff: float | None = None,
    term_limit: int | None = None,
    dump_force: Path | None = None,
    verify_small_cell: bool = False,
) -> dict[str, Any]:
    chosen, env = build_pinning_env(threads, cpus)
    cpu_list = ",".join(str(c) for c in chosen)
    cmd = [
        "taskset",
        "-c",
        cpu_list,
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
        "--eam-table-path",
        "CppCore/tests/data/eam_table/Cu_u3.eam",
    ]
    if cutoff is not None:
        cmd += ["--cutoff", f"{cutoff:.12g}"]
    if term_limit is not None:
        cmd += ["--term-limit", str(term_limit)]
    if dump_force is not None:
        cmd += ["--dump-force", str(dump_force)]
    if verify_small_cell:
        cmd += ["--verify-small-cell"]

    allowed_set = set(chosen)

    def _pin_child() -> None:
        try:
            os.sched_setaffinity(0, allowed_set)
        except (AttributeError, OSError):
            pass

    proc = run_cmd(cmd, cwd=ctx.worktree, env=env, preexec_fn=_pin_child)
    payload = json.loads(proc.stdout)

    observed_par = int(payload["omp_parallel_threads_per_sample"][0])
    if observed_par != threads:
        raise RuntimeError(
            f"thread count mismatch for {ctx.pr}/{ctx.arm}/{potential}: "
            f"expected {threads}, got {observed_par}"
        )
    observed_aff = str(payload["affinity_per_sample"][0])
    aff_cpus = {int(part) for part in observed_aff.split(",") if part.strip()}
    if not aff_cpus or not aff_cpus.issubset(allowed_set):
        raise RuntimeError(
            f"affinity mismatch for {ctx.pr}/{ctx.arm}: {aff_cpus} not subset of {allowed_set}"
        )
    if threads == 1 and aff_cpus != allowed_set:
        raise RuntimeError(f"1-thread affinity expected {allowed_set}, got {aff_cpus}")
    par_aff_raw = str(payload["omp_parallel_affinity_per_sample"][0])
    for mask in par_aff_raw.split(";"):
        t_cpus = {int(part) for part in mask.split(",") if part.strip()}
        if not t_cpus or not t_cpus.issubset(allowed_set):
            raise RuntimeError(
                f"parallel thread affinity {t_cpus} not subset of {allowed_set}"
            )
        if threads == 1 and t_cpus != allowed_set:
            raise RuntimeError(
                f"1-thread parallel affinity expected {allowed_set}, got {t_cpus}"
            )
    return payload


def worker_probe_failure(
    ctx: ArmContext,
    *,
    potential: str,
    atoms: int,
    threads: int,
    sample_seconds: float,
    cpus: list[int],
) -> str | None:
    try:
        run_worker_once(
            ctx,
            potential=potential,
            atoms=atoms,
            threads=threads,
            sample_seconds=sample_seconds,
            cpus=cpus,
        )
    except subprocess.CalledProcessError as err:
        return " ".join((err.stderr or err.stdout or "failed").split())
    return None


def parse_force_dump(path: Path) -> tuple[float, list[float]]:
    import array

    with path.open("rb") as handle:
        raw_n = handle.read(8)
        if len(raw_n) != 8:
            raise RuntimeError("truncated force dump header")
        n_atoms = int.from_bytes(raw_n, byteorder="little", signed=False)
        e_bytes = handle.read(8)
        if len(e_bytes) != 8:
            raise RuntimeError("truncated force dump energy")
        e_arr = array.array("d")
        e_arr.frombytes(e_bytes)
        f_arr = array.array("d")
        f_arr.fromfile(handle, 3 * n_atoms)
    return float(e_arr[0]), list(f_arr)


def compare_force_dumps(path_a: Path, path_b: Path) -> dict[str, Any]:
    e_a, f_a = parse_force_dump(path_a)
    e_b, f_b = parse_force_dump(path_b)
    if len(f_a) != len(f_b):
        raise RuntimeError("force vector length mismatch")
    max_f = 0.0
    bit_f = True
    for va, vb in zip(f_a, f_b):
        if struct.pack("<d", va) != struct.pack("<d", vb):
            bit_f = False
        d = abs(va - vb)
        if d > max_f:
            max_f = d
    bit_e = struct.pack("<d", e_a) == struct.pack("<d", e_b)
    return {
        "energy_bit_identical": bit_e,
        "force_bit_identical": bit_f,
        "max_abs_energy_diff": abs(e_a - e_b),
        "max_abs_force_diff": max_f,
    }


def load_raw_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def append_raw_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_CSV_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in RAW_CSV_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(row["pr"]),
        str(row["cell_id"]),
        int(row["round_idx"]),
        int(row["slot_idx"]),
        str(row["arm"]),
    )


def execute_cell_rounds(
    *,
    cell: CellSpec,
    arms_map: dict[str, ArmContext],
    schedule: list[tuple[str, ...]],
    start_round: int,
    end_round: int,
    out_csv: Path,
    seen_keys: set[tuple[str, str, int, int, str]],
    cpus: list[int],
    sample_seconds: float,
    log_handle: Any,
) -> None:
    for r_idx in range(start_round, end_round):
        order = schedule[r_idx] if cell.cell_kind == "comparative" else ("head",)
        for slot_idx, arm_name in enumerate(order):
            key = (cell.pr, cell.cell_id, r_idx, slot_idx, arm_name)
            if key in seen_keys:
                continue
            ctx = arms_map[arm_name]
            res = run_worker_once(
                ctx,
                potential=cell.potential,
                atoms=cell.atoms,
                threads=cell.threads,
                sample_seconds=sample_seconds,
                cpus=cpus,
                cutoff=cell.cutoff,
                term_limit=cell.term_limit,
                verify_small_cell=False,
            )
            record = {
                "pr": cell.pr,
                "cell_id": cell.cell_id,
                "cell_kind": cell.cell_kind,
                "potential": cell.potential,
                "atoms": cell.atoms,
                "threads": cell.threads,
                "round_idx": r_idx,
                "slot_idx": slot_idx,
                "arm": arm_name,
                "commit_sha": ctx.commit_sha,
                "tree_sha": ctx.tree_sha,
                "total_ns_per_eval": f"{float(res['total_ns_per_eval'][0]):.12g}",
                "evals": int(res["evals_per_sample"][0]),
                "omp_max_threads": int(res["omp_max_threads_per_sample"][0]),
                "omp_parallel_threads": int(res["omp_parallel_threads_per_sample"][0]),
                "affinity": str(res["affinity_per_sample"][0]),
                "omp_parallel_affinity": str(res["omp_parallel_affinity_per_sample"][0]),
                "energy": f"{float(res['energy']):.17g}",
                "energy_bits": str(res["energy_bits"]),
                "force_hash": str(res["force_hash"]),
                "list_reused": str(bool(res["list_reused"])).lower(),
                "repeat_energy_bit_identical": str(
                    bool(res["repeat_energy_bit_identical"])
                ).lower(),
                "repeat_force_bit_identical": str(
                    bool(res["repeat_force_bit_identical"])
                ).lower(),
                "repeat_max_abs_force_diff": f"{float(res['repeat_max_abs_force_diff']):.17g}",
                "small_cell_regime": str(
                    bool(res["small_cell_multi_image_regime"])
                ).lower(),
            }
            append_raw_row(out_csv, record)
            seen_keys.add(key)
            log_handle.write(
                f"{cell.pr} {cell.cell_id} r={r_idx} s={slot_idx} arm={arm_name} "
                f"ns={record['total_ns_per_eval']} evals={record['evals']}\n"
            )
            log_handle.flush()


def collect_samples_by_arm(
    rows: list[dict[str, str]], pr: str, cell_id: str, max_round: int | None = None
) -> dict[str, list[float]]:
    grouped: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    for row in rows:
        if row["pr"] != pr or row["cell_id"] != cell_id:
            continue
        r_idx = int(row["round_idx"])
        if max_round is not None and r_idx >= max_round:
            continue
        grouped[row["arm"]][r_idx] = float(row["total_ns_per_eval"])
    return {
        arm: [vals[r] for r in sorted(vals.keys())]
        for arm, vals in grouped.items()
    }


def compute_cell_pilot_plan(
    cell: CellSpec, rows: list[dict[str, str]], seed_base: int
) -> dict[str, Any]:
    by_arm = collect_samples_by_arm(rows, cell.pr, cell.cell_id, max_round=PILOT_ROUNDS)
    if cell.cell_kind == "comparative":
        pilot_stats = bootstrap_ratio_ci(
            by_arm["head"][:PILOT_ROUNDS],
            by_arm["control"][:PILOT_ROUNDS],
            resamples=BOOTSTRAP_PILOT,
            seed=seed_base,
        )
        rel_se = pilot_stats["rel_se"]
    else:
        rel_se = bootstrap_single_rel_se(
            by_arm["head"][:PILOT_ROUNDS],
            resamples=BOOTSTRAP_PILOT,
            seed=seed_base,
        )
    n_needed = compute_n_needed(rel_se)
    return {
        "pilot_rounds": PILOT_ROUNDS,
        "pilot_rel_se": rel_se,
        "n_needed": n_needed,
    }


def analyze_all(
    *,
    raw_rows: list[dict[str, str]],
    arms_meta: dict[str, Any],
    pilot_plan: dict[str, Any],
    small_cell_checks: list[dict[str, Any]],
    pr293_agreements: list[dict[str, Any]],
    pr293_term_limit_checks: list[dict[str, Any]],
    machine_meta: dict[str, Any],
) -> dict[str, Any]:
    cells = all_cells()
    pr_summaries: dict[str, Any] = {}

    for pr_key, spec in PR_SPECS.items():
        pr_cells = [c for c in cells if c.pr == pr_key]
        comp_cells = [c for c in pr_cells if c.cell_kind == "comparative"]
        n_comp = len(comp_cells)
        comp_results: list[dict[str, Any]] = []
        small_results: list[dict[str, Any]] = []

        for cell in comp_cells:
            plan = pilot_plan[f"{cell.pr}:{cell.cell_id}"]
            n_rounds = int(plan["n_needed"])
            by_arm = collect_samples_by_arm(raw_rows, cell.pr, cell.cell_id, max_round=n_rounds)
            ctrl = by_arm["control"]
            ctrl_aa = by_arm["control_aa"]
            head = by_arm["head"]

            seed = abs(hash((cell.pr, cell.cell_id))) % (2**31)
            ratio_stats = bootstrap_ratio_ci(
                head, ctrl, resamples=BOOTSTRAP_FINAL, seed=seed + 1
            )
            aa_stats = symmetrized_aa_band(
                ctrl_aa, ctrl, resamples=BOOTSTRAP_FINAL, seed=seed + 2
            )
            mw = exact_mann_whitney_two_sided(head, ctrl)
            bonf_p = min(1.0, mw["p_two_sided"] * float(n_comp))
            verdict = classify_cell_verdict(
                ratio_ci_low=ratio_stats["ci_low"],
                ratio_ci_high=ratio_stats["ci_high"],
                aa_band_low=aa_stats["band_low"],
                aa_band_high=aa_stats["band_high"],
                bonferroni_p=bonf_p,
            )

            comp_results.append(
                {
                    "cell_id": cell.cell_id,
                    "potential": cell.potential,
                    "atoms": cell.atoms,
                    "threads": cell.threads,
                    "n_rounds": len(ctrl),
                    "pilot_rel_se": plan["pilot_rel_se"],
                    "achieved_rel_se": ratio_stats["rel_se"],
                    "control_median_ns": statistics.median(ctrl),
                    "control_iqr_ns": iqr(ctrl),
                    "control_aa_median_ns": statistics.median(ctrl_aa),
                    "control_aa_iqr_ns": iqr(ctrl_aa),
                    "head_median_ns": statistics.median(head),
                    "head_iqr_ns": iqr(head),
                    "ratio": ratio_stats["ratio"],
                    "ratio_ci_low": ratio_stats["ci_low"],
                    "ratio_ci_high": ratio_stats["ci_high"],
                    "aa_ratio": aa_stats["aa_ratio"],
                    "aa_band_low": aa_stats["band_low"],
                    "aa_band_high": aa_stats["band_high"],
                    "aa_per_round_min": aa_stats["per_round_min"],
                    "aa_per_round_max": aa_stats["per_round_max"],
                    "mann_whitney_u": mw["u"],
                    "exact_mann_whitney_p": mw["p_two_sided"],
                    "bonferroni_p": bonf_p,
                    "verdict": verdict,
                }
            )

        for cell in pr_cells:
            if cell.cell_kind != "small_head_only":
                continue
            plan = pilot_plan[f"{cell.pr}:{cell.cell_id}"]
            n_rounds = int(plan["n_needed"])
            by_arm = collect_samples_by_arm(raw_rows, cell.pr, cell.cell_id, max_round=n_rounds)
            head = by_arm["head"]
            med = statistics.median(head)
            small_results.append(
                {
                    "cell_id": cell.cell_id,
                    "potential": cell.potential,
                    "atoms": cell.atoms,
                    "threads": cell.threads,
                    "n_rounds": len(head),
                    "pilot_rel_se": plan["pilot_rel_se"],
                    "head_median_ns": med,
                    "head_iqr_ns": iqr(head),
                    "head_ns_per_atom": med / float(cell.atoms),
                    "head_eval_per_s": 1.0e9 / med if med > 0.0 else 0.0,
                }
            )

        overall = aggregate_pr_verdict([r["verdict"] for r in comp_results])
        pr_summaries[pr_key] = {
            "pr_number": spec["pr_number"],
            "title": spec["title"],
            "n_comparative_cells": n_comp,
            "overall_verdict": overall,
            "counts": {
                "regression": sum(1 for r in comp_results if r["verdict"] == "regression"),
                "win": sum(1 for r in comp_results if r["verdict"] == "win"),
                "neutral": sum(1 for r in comp_results if r["verdict"] == "neutral"),
            },
            "comparative_cells": comp_results,
            "small_head_only_cells": small_results,
        }

    # Count (slot, arm) pairs across comparative raw rows to verify alternation.
    slot_arm_counts: dict[str, dict[str, int]] = {}
    for row in raw_rows:
        if row["cell_kind"] != "comparative":
            continue
        pr = row["pr"]
        key = f"{pr}:slot{row['slot_idx']}:{row['arm']}"
        slot_arm_counts.setdefault(pr, {})[key] = slot_arm_counts.get(pr, {}).get(key, 0) + 1

    return {
        "base_develop_sha": BASE_DEVELOP_SHA,
        "machine": machine_meta,
        "arms": arms_meta,
        "pilot_plan": pilot_plan,
        "slot_arm_counts": slot_arm_counts,
        "prs": pr_summaries,
        "small_cell_checks": small_cell_checks,
        "pr293_agreements": pr293_agreements,
        "pr293_term_limit_checks": pr293_term_limit_checks,
    }


def fmt(val: float, digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "nan"
    return f"{val:.{digits}g}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Pre-landing Performance Report: PR #287 and PR #293")
    lines.append("")
    lines.append("## Environment and Protocol")
    lines.append("")
    m = summary["machine"]
    lines.append(f"- Base develop commit: `{summary['base_develop_sha']}`")
    lines.append(f"- Compiler: `{m['compiler_version']}`")
    lines.append(f"- Build configuration: `{m['build_type']}`")
    lines.append(f"- CPU model: `{m['cpu_model']}` (`{m['physical_cores']}` physical cores)")
    lines.append(
        "- Pinning (1 thread): `taskset` to 1 core, all thread env vars set to `1`, "
        "`OMP_PROC_BIND=close`, child affinity verified"
    )
    lines.append(
        "- Pinning (multi-thread): `taskset` to exact core count (`2/4/8`), "
        "`OMP_NUM_THREADS=T`, `OMP_PROC_BIND=close`, explicit `OMP_PLACES`"
    )
    lines.append(
        f"- Sampling: pilot `{PILOT_ROUNDS}` rounds, adaptive `n` up to `{MAX_ROUNDS}` "
        f"targeting relative SE <= `{TARGET_REL_SE:.0%}` on `head/control` ratio; "
        f"no `--quick`, no warm-up discards beyond bench's own, no rescue runs"
    )
    lines.append(
        "- Decision rule: `regression` if 95% bootstrap CI of `head/control` lies strictly above "
        "the A/A band and exact Mann-Whitney `p < 0.01` after Bonferroni across PR cells; "
        "`win` by mirror rule; otherwise `neutral`"
    )
    lines.append("")

    lines.append("## Built Arms (Commit and Tree SHAs)")
    lines.append("")
    lines.append("|PR|Arm|Commit SHA|Tree SHA|No `-march=native`|No `-ffast-math`|")
    lines.append("|---|---|---|---|---|---|")
    for pr_key in ("pr287", "pr293"):
        for arm in ARMS:
            info = summary["arms"][pr_key][arm]
            audit = info["flag_audit"]
            lines.append(
                f"|#{summary['prs'][pr_key]['pr_number']}|{arm}|`{info['commit_sha']}`|"
                f"`{info['tree_sha']}`|{not audit['has_march_native']}|{not audit['has_ffast_math']}|"
            )
    lines.append("")

    lines.append("## Per-PR Verdict Summary")
    lines.append("")
    lines.append("|PR|Title|Cells|Wins|Regressions|Neutral|Verdict|")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for pr_key in ("pr287", "pr293"):
        pr = summary["prs"][pr_key]
        c = pr["counts"]
        lines.append(
            f"|#{pr['pr_number']}|{pr['title']}|{pr['n_comparative_cells']}|"
            f"{c['win']}|{c['regression']}|{c['neutral']}|**{pr['overall_verdict']}**|"
        )
    lines.append("")

    # PR #287 section
    pr287 = summary["prs"]["pr287"]
    lines.append("## PR #287: Large-cell Comparative Results (`head` vs `control` and `control_aa`)")
    lines.append("")
    lines.append(
        "|Potential|Atoms|Rounds|Control Med (ns)|Control IQR (ns)|A/A Med (ns)|"
        "Head Med (ns)|Head IQR (ns)|Head/Ctrl Ratio [95% CI]|A/A Band|"
        "Exact MW p|Bonf p|Verdict|"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|"
    )
    for r in pr287["comparative_cells"]:
        lines.append(
            f"|{r['potential']}|{r['atoms']}|{r['n_rounds']}|"
            f"{fmt(r['control_median_ns'], 5)}|{fmt(r['control_iqr_ns'], 4)}|"
            f"{fmt(r['control_aa_median_ns'], 5)}|"
            f"{fmt(r['head_median_ns'], 5)}|{fmt(r['head_iqr_ns'], 4)}|"
            f"{fmt(r['ratio'], 4)} [{fmt(r['ratio_ci_low'], 4)}, {fmt(r['ratio_ci_high'], 4)}]|"
            f"[{fmt(r['aa_band_low'], 4)}, {fmt(r['aa_band_high'], 4)}]|"
            f"{fmt(r['exact_mann_whitney_p'], 4)}|{fmt(r['bonferroni_p'], 4)}|{r['verdict']}|"
        )
    lines.append("")

    # Small cell timings & cost
    eam108 = next(
        (r for r in pr287["comparative_cells"] if r["potential"] == "eam_table" and r["atoms"] == 108),
        None,
    )
    ref_ns_per_atom = (eam108["head_median_ns"] / 108.0) if eam108 else None
    lines.append("## PR #287: Small-cell Multi-image Cost (`EAMTablePot`, control rejects `edge <= 2*cutoff`)")
    lines.append("")
    lines.append(
        "|Cell|Atoms|Rounds|Head Med (ns)|Head IQR (ns)|Head ns/atom|Cost vs 108-atom ns/atom|"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in pr287["small_head_only_cells"]:
        rel = (
            f"{fmt(r['head_ns_per_atom'] / ref_ns_per_atom, 4)}x"
            if ref_ns_per_atom and ref_ns_per_atom > 0.0
            else "n/a"
        )
        lines.append(
            f"|{r['potential']}|{r['atoms']}|{r['n_rounds']}|"
            f"{fmt(r['head_median_ns'], 5)}|{fmt(r['head_iqr_ns'], 4)}|"
            f"{fmt(r['head_ns_per_atom'], 5)}|{rel}|"
        )
    lines.append("")

    lines.append("## PR #287: Small-cell Correctness Verification")
    lines.append("")
    lines.append(
        "|Potential|Atoms|Min Face / (2*Cut)|Control Status|Reference|Max |dE||Max |dF||Pairs (Got/Ref)|Notes|"
    )
    lines.append("|---|---:|---|---|---|---|---|---|---|")
    for chk in summary["small_cell_checks"]:
        ratio_fc = chk["min_face_spacing"] / (2.0 * chk["cutoff"]) if chk["cutoff"] > 0 else 0.0
        lines.append(
            f"|{chk['potential']}|{chk['atoms']}|{fmt(ratio_fc, 3)}|"
            f"{chk['control_status']}|{chk['reference_kind']}|"
            f"{fmt(chk['max_abs_energy_diff'], 4)}|{fmt(chk['max_abs_force_diff'], 4)}|"
            f"{chk['got_pairs']}/{chk['ref_pairs']}|{chk['note']}|"
        )
    lines.append("")

    # PR #293 section
    pr293 = summary["prs"]["pr293"]
    lines.append("## PR #293: Sutton-Chen Comparative Results (`head` vs `control` and `control_aa`)")
    lines.append("")
    lines.append(
        "|Atoms|Threads|Rounds|Control Med (ns)|Control IQR (ns)|A/A Med (ns)|"
        "Head Med (ns)|Head IQR (ns)|Head/Ctrl Ratio [95% CI]|A/A Band|"
        "Exact MW p|Bonf p|Verdict|"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|"
    )
    for r in pr293["comparative_cells"]:
        lines.append(
            f"|{r['atoms']}|{r['threads']}|{r['n_rounds']}|"
            f"{fmt(r['control_median_ns'], 5)}|{fmt(r['control_iqr_ns'], 4)}|"
            f"{fmt(r['control_aa_median_ns'], 5)}|"
            f"{fmt(r['head_median_ns'], 5)}|{fmt(r['head_iqr_ns'], 4)}|"
            f"{fmt(r['ratio'], 4)} [{fmt(r['ratio_ci_low'], 4)}, {fmt(r['ratio_ci_high'], 4)}]|"
            f"[{fmt(r['aa_band_low'], 4)}, {fmt(r['aa_band_high'], 4)}]|"
            f"{fmt(r['exact_mann_whitney_p'], 4)}|{fmt(r['bonferroni_p'], 4)}|{r['verdict']}|"
        )
    lines.append("")

    lines.append("## PR #293: Force Agreement vs Control and Determinism")
    lines.append("")
    lines.append(
        "|Atoms|Threads|Head vs Control Bitwise|Head vs Control Max |dF||"
        "Head vs T=1 Bitwise|Head vs T=1 Max |dF||Repeat-Call Bitwise|Repeat-Call Max |dF||"
    )
    lines.append("|---:|---:|---|---:|---|---:|---|---:|")
    for ag in summary["pr293_agreements"]:
        lines.append(
            f"|{ag['atoms']}|{ag['threads']}|"
            f"{'yes' if ag['vs_control_force_bit_identical'] else 'no'}|"
            f"{fmt(ag['vs_control_max_abs_force_diff'], 4)}|"
            f"{'yes' if ag['vs_t1_force_bit_identical'] else 'no'}|"
            f"{fmt(ag['vs_t1_max_abs_force_diff'], 4)}|"
            f"{'yes' if ag['repeat_force_bit_identical'] else 'no'}|"
            f"{fmt(ag['repeat_max_abs_force_diff'], 4)}|"
        )
    lines.append("")

    if summary["pr293_term_limit_checks"]:
        lines.append("## PR #293: Finite Term-Limit Fallback Check (`atoms=10976, term_limit=10000`)")
        lines.append("")
        lines.append(
            "|Threads|Head Med (ns)|Control Med (ns)|Head/Ctrl Ratio|Head vs Control Bitwise|Max |dF||"
        )
        lines.append("|---:|---:|---:|---:|---|---:|")
        for tl in summary["pr293_term_limit_checks"]:
            lines.append(
                f"|{tl['threads']}|{fmt(tl['head_ns'], 5)}|{fmt(tl['control_ns'], 5)}|"
                f"{fmt(tl['head_ns'] / tl['control_ns'], 4)}|"
                f"{'yes' if tl['force_bit_identical'] else 'no'}|"
                f"{fmt(tl['max_abs_force_diff'], 4)}|"
            )
        lines.append("")

    lines.append("## Slot Alternation Audit `(slot, arm)` Counts")
    lines.append("")
    lines.append("|PR|Slot 0 (ctrl / aa / head)|Slot 1 (ctrl / aa / head)|Slot 2 (ctrl / aa / head)|")
    lines.append("|---|---|---|---|")
    for pr_key in ("pr287", "pr293"):
        counts = summary["slot_arm_counts"].get(pr_key, {})
        parts = []
        for slot in (0, 1, 2):
            c0 = counts.get(f"{pr_key}:slot{slot}:control", 0)
            c1 = counts.get(f"{pr_key}:slot{slot}:control_aa", 0)
            c2 = counts.get(f"{pr_key}:slot{slot}:head", 0)
            parts.append(f"{c0} / {c1} / {c2}")
        lines.append(f"|#{summary['prs'][pr_key]['pr_number']}|{parts[0]}|{parts[1]}|{parts[2]}|")
    lines.append("")

    return "\n".join(lines)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def compiler_version(cxx: str) -> str:
    return run_cmd([cxx, "--version"]).stdout.splitlines()[0].strip()


def run_small_cell_checks(
    pr287_arms: dict[str, ArmContext], cpus: list[int], sample_seconds: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    targets = [
        ("eam_table", 4),
        ("eam_table", 8),
        ("eam_table", 16),
        ("eam_table", 32),
        ("eam_table_tri3", 3),
        ("nlist_build", 32),
        ("lj", 32),
        ("morse", 32),
        ("eam", 32),
    ]
    for pot, atoms in targets:
        ctrl_err = worker_probe_failure(
            pr287_arms["control"],
            potential=pot,
            atoms=atoms,
            threads=1,
            sample_seconds=sample_seconds,
            cpus=cpus,
        )
        ctrl_status = "ok" if ctrl_err is None else f"rejected: {ctrl_err}"
        head_res = run_worker_once(
            pr287_arms["head"],
            potential=pot,
            atoms=atoms,
            threads=1,
            sample_seconds=sample_seconds,
            cpus=cpus,
            verify_small_cell=True,
        )
        corr = head_res["correctness"]
        note = str(corr["note"])
        ref_kind = str(corr["reference_kind"])
        max_e = float(corr["max_abs_energy_diff"])
        max_f = float(corr["max_abs_force_diff"])

        if pot == "eam" and ctrl_err is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                dc = Path(tmpdir) / "c.bin"
                dh = Path(tmpdir) / "h.bin"
                run_worker_once(
                    pr287_arms["control"],
                    potential=pot,
                    atoms=atoms,
                    threads=1,
                    sample_seconds=sample_seconds,
                    cpus=cpus,
                    dump_force=dc,
                )
                run_worker_once(
                    pr287_arms["head"],
                    potential=pot,
                    atoms=atoms,
                    threads=1,
                    sample_seconds=sample_seconds,
                    cpus=cpus,
                    dump_force=dh,
                )
                diff = compare_force_dumps(dc, dh)
                ref_kind = "head_vs_control_minimage"
                max_e = float(diff["max_abs_energy_diff"])
                max_f = float(diff["max_abs_force_diff"])
                note = "bitwise_identical" if diff["force_bit_identical"] else "differs"

        checks.append(
            {
                "potential": pot,
                "atoms": atoms,
                "cutoff": float(head_res["cutoff"]),
                "min_face_spacing": float(head_res["min_face_spacing"]),
                "control_status": ctrl_status,
                "reference_kind": ref_kind,
                "max_abs_energy_diff": max_e,
                "max_abs_force_diff": max_f,
                "got_pairs": int(corr["got_pairs"]),
                "ref_pairs": int(corr["ref_pairs"]),
                "note": note,
            }
        )
    return checks


def run_pr293_agreements(
    pr293_arms: dict[str, ArmContext], cpus: list[int], sample_seconds: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agreements: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for atoms in (256, 1024, 4096, 10976):
            t1_head_dump = tmp / f"head_n{atoms}_t1.bin"
            for threads in (1, 2, 4, 8):
                c_dump = tmp / f"ctrl_n{atoms}_t{threads}.bin"
                h_dump = tmp / f"head_n{atoms}_t{threads}.bin"
                h2_dump = tmp / f"head2_n{atoms}_t{threads}.bin"
                run_worker_once(
                    pr293_arms["control"],
                    potential="sutton-chen",
                    atoms=atoms,
                    threads=threads,
                    sample_seconds=sample_seconds,
                    cpus=cpus,
                    dump_force=c_dump,
                )
                res1 = run_worker_once(
                    pr293_arms["head"],
                    potential="sutton-chen",
                    atoms=atoms,
                    threads=threads,
                    sample_seconds=sample_seconds,
                    cpus=cpus,
                    dump_force=h_dump,
                )
                run_worker_once(
                    pr293_arms["head"],
                    potential="sutton-chen",
                    atoms=atoms,
                    threads=threads,
                    sample_seconds=sample_seconds,
                    cpus=cpus,
                    dump_force=h2_dump,
                )
                if threads == 1:
                    t1_head_dump = h_dump
                vs_ctrl = compare_force_dumps(h_dump, c_dump)
                vs_t1 = compare_force_dumps(h_dump, t1_head_dump)
                vs_rep = compare_force_dumps(h_dump, h2_dump)
                agreements.append(
                    {
                        "atoms": atoms,
                        "threads": threads,
                        "vs_control_energy_bit_identical": vs_ctrl["energy_bit_identical"],
                        "vs_control_force_bit_identical": vs_ctrl["force_bit_identical"],
                        "vs_control_max_abs_energy_diff": vs_ctrl["max_abs_energy_diff"],
                        "vs_control_max_abs_force_diff": vs_ctrl["max_abs_force_diff"],
                        "vs_t1_energy_bit_identical": vs_t1["energy_bit_identical"],
                        "vs_t1_force_bit_identical": vs_t1["force_bit_identical"],
                        "vs_t1_max_abs_force_diff": vs_t1["max_abs_force_diff"],
                        "repeat_force_bit_identical": (
                            vs_rep["force_bit_identical"]
                            and bool(res1["repeat_force_bit_identical"])
                        ),
                        "repeat_max_abs_force_diff": max(
                            float(vs_rep["max_abs_force_diff"]),
                            float(res1["repeat_max_abs_force_diff"]),
                        ),
                    }
                )

    term_checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for threads in (1, 2, 4, 8):
            c_dump = tmp / f"tl_ctrl_t{threads}.bin"
            h_dump = tmp / f"tl_head_t{threads}.bin"
            c_res = run_worker_once(
                pr293_arms["control"],
                potential="sutton-chen",
                atoms=10976,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus,
                term_limit=10000,
                dump_force=c_dump,
            )
            h_res = run_worker_once(
                pr293_arms["head"],
                potential="sutton-chen",
                atoms=10976,
                threads=threads,
                sample_seconds=sample_seconds,
                cpus=cpus,
                term_limit=10000,
                dump_force=h_dump,
            )
            diff = compare_force_dumps(h_dump, c_dump)
            term_checks.append(
                {
                    "threads": threads,
                    "control_ns": float(c_res["total_ns_per_eval"][0]),
                    "head_ns": float(h_res["total_ns_per_eval"][0]),
                    "force_bit_identical": diff["force_bit_identical"],
                    "max_abs_force_diff": diff["max_abs_force_diff"],
                }
            )
    return agreements, term_checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument(
        "--worktree-root",
        default=str(Path(tempfile.gettempdir()) / "rgpot-perf-worktrees"),
    )
    parser.add_argument(
        "--build-root",
        default=str(Path(tempfile.gettempdir()) / "rgpot-perf-builds"),
    )
    parser.add_argument(
        "--log-dir",
        default=str(Path(tempfile.gettempdir()) / "rgpot-perf-logs"),
    )
    parser.add_argument("--out-csv", default="bench/results/pr287_293_raw.csv")
    parser.add_argument("--out-json", default="bench/results/pr287_293_summary.json")
    parser.add_argument("--out-md", default="bench/results/pr287_293_report.md")
    parser.add_argument("--sample-seconds", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--cc", default="gcc")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate report markdown from existing out-json without running benchmarks.",
    )
    args = parser.parse_args()

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        summary = json.loads(out_json.read_text(encoding="utf-8"))
        out_md.write_text(render_markdown(summary) + "\n", encoding="utf-8")
        return 0

    repo = Path(args.repo).resolve()
    worktree_root = Path(args.worktree_root).resolve()
    build_root = Path(args.build_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cpus = exclusive_cpus()
    if len(cpus) < 8:
        raise RuntimeError(f"need at least 8 physical cores, found {cpus}")

    worker_source = repo / "bench" / "rgpot_perf_worker.cc"
    arms_by_pr: dict[str, dict[str, ArmContext]] = {}
    arms_meta: dict[str, dict[str, Any]] = {}

    for pr_key, spec in PR_SPECS.items():
        arms_by_pr[pr_key] = {}
        arms_meta[pr_key] = {}
        for arm in ARMS:
            sha = spec["head_sha"] if arm == "head" else spec["control_sha"]
            ctx = build_arm(
                main_repo=repo,
                pr=pr_key,
                arm=arm,
                expected_sha=sha,
                worktree_root=worktree_root,
                build_root=build_root,
                worker_source=worker_source,
                cc=args.cc,
                cxx=args.cxx,
            )
            arms_by_pr[pr_key][arm] = ctx
            arms_meta[pr_key][arm] = {
                "commit_sha": ctx.commit_sha,
                "tree_sha": ctx.tree_sha,
                "flag_audit": audit_build_flags(ctx.build_dir),
            }

    existing_rows = load_raw_csv(out_csv)
    seen_keys = {row_key(r) for r in existing_rows}

    plan_path = log_dir / "pilot_plan.json"
    if plan_path.is_file():
        pilot_plan: dict[str, Any] = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        pilot_plan = {}

    run_log_path = log_dir / "runner.log"
    with run_log_path.open("a", encoding="utf-8") as log_handle:
        cells = all_cells()
        # Phase 1: 14-round pilot
        for idx, cell in enumerate(cells):
            cell_seed = args.seed + idx * 101
            schedule = make_round_schedule(MAX_ROUNDS, cell_seed)
            execute_cell_rounds(
                cell=cell,
                arms_map=arms_by_pr[cell.pr],
                schedule=schedule,
                start_round=0,
                end_round=PILOT_ROUNDS,
                out_csv=out_csv,
                seen_keys=seen_keys,
                cpus=cpus,
                sample_seconds=args.sample_seconds,
                log_handle=log_handle,
            )

        raw_rows = load_raw_csv(out_csv)
        for idx, cell in enumerate(cells):
            plan_key = f"{cell.pr}:{cell.cell_id}"
            if plan_key not in pilot_plan:
                pilot_plan[plan_key] = compute_cell_pilot_plan(
                    cell, raw_rows, args.seed + idx * 101 + 7
                )
        plan_path.write_text(
            json.dumps(pilot_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Phase 2: adaptive rounds up to n_needed (capped at 48)
        for idx, cell in enumerate(cells):
            plan_key = f"{cell.pr}:{cell.cell_id}"
            n_needed = int(pilot_plan[plan_key]["n_needed"])
            if n_needed <= PILOT_ROUNDS:
                continue
            cell_seed = args.seed + idx * 101
            schedule = make_round_schedule(MAX_ROUNDS, cell_seed)
            execute_cell_rounds(
                cell=cell,
                arms_map=arms_by_pr[cell.pr],
                schedule=schedule,
                start_round=PILOT_ROUNDS,
                end_round=n_needed,
                out_csv=out_csv,
                seen_keys=seen_keys,
                cpus=cpus,
                sample_seconds=args.sample_seconds,
                log_handle=log_handle,
            )

    raw_rows = load_raw_csv(out_csv)
    small_checks = run_small_cell_checks(arms_by_pr["pr287"], cpus, args.sample_seconds)
    pr293_agreements, pr293_term_checks = run_pr293_agreements(
        arms_by_pr["pr293"], cpus, args.sample_seconds
    )

    machine_meta = {
        "cpu_model": cpu_model(),
        "physical_cores": len(cpus),
        "compiler_version": compiler_version(args.cxx),
        "build_type": "release (meson -Dbuildtype=release -Dwith_eam_table=true -Dwith_fast_nlist=enabled)",
        "sample_seconds": args.sample_seconds,
    }

    summary = analyze_all(
        raw_rows=raw_rows,
        arms_meta=arms_meta,
        pilot_plan=pilot_plan,
        small_cell_checks=small_checks,
        pr293_agreements=pr293_agreements,
        pr293_term_limit_checks=pr293_term_checks,
        machine_meta=machine_meta,
    )
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(summary) + "\n", encoding="utf-8")

    (log_dir / "small_cell_checks.json").write_text(
        json.dumps(small_checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (log_dir / "pr293_agreements.json").write_text(
        json.dumps(pr293_agreements, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
