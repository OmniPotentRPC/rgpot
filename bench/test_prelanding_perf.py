#!/usr/bin/env python3
"""Unit tests for the pre-landing performance harness."""

from __future__ import annotations

import itertools
import math
import platform
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from bench.run_prelanding_perf import (
    ARMS,
    MAX_ROUNDS,
    PILOT_ROUNDS,
    SINGLE_THREAD_ENV_VARS,
    aggregate_pr_verdict,
    append_raw_row,
    bootstrap_ratio_ci,
    build_pinning_env,
    classify_cell_verdict,
    compute_n_needed,
    exact_mann_whitney_two_sided,
    load_raw_csv,
    make_round_schedule,
    render_markdown,
    row_key,
    symmetrized_aa_band,
)


class TestAlternatingSchedule(unittest.TestCase):
    def test_slot_arm_pairs_are_genuinely_alternated(self) -> None:
        for n_rounds in (14, 18, 24, 48):
            schedule = make_round_schedule(n_rounds, seed=20261119)
            self.assertEqual(len(schedule), n_rounds)
            pair_counts: Counter[tuple[int, str]] = Counter()
            for order in schedule:
                self.assertEqual(sorted(order), sorted(ARMS))
                for slot_idx, arm in enumerate(order):
                    pair_counts[(slot_idx, arm)] += 1

            min_expected = n_rounds // len(ARMS)
            for slot_idx in range(len(ARMS)):
                for arm in ARMS:
                    cnt = pair_counts[(slot_idx, arm)]
                    self.assertGreaterEqual(
                        cnt,
                        min_expected,
                        f"slot={slot_idx} arm={arm} under-represented for n={n_rounds}",
                    )
            if n_rounds % 6 == 0:
                expected = n_rounds // 3
                for slot_idx in range(len(ARMS)):
                    for arm in ARMS:
                        self.assertEqual(pair_counts[(slot_idx, arm)], expected)

            # Verify multiple distinct permutations appear across rounds.
            self.assertGreater(len(set(schedule)), 1)


class TestPilotAndStats(unittest.TestCase):
    def test_n_needed_pilot_and_cap(self) -> None:
        self.assertEqual(compute_n_needed(0.01), PILOT_ROUNDS)
        self.assertEqual(compute_n_needed(0.03), PILOT_ROUNDS)
        # 14 * (0.045 / 0.03)^2 = 14 * 2.25 = 31.5 -> 32
        self.assertEqual(compute_n_needed(0.045), 32)
        self.assertEqual(compute_n_needed(0.20), MAX_ROUNDS)

    def test_exact_mann_whitney_untied_and_tied(self) -> None:
        # Complete separation for n=4 vs n=4: U = 0 or 16.
        # Two-sided p = 2 / C(8, 4) = 2 / 70 = 1/35.
        res_sep = exact_mann_whitney_two_sided([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
        self.assertAlmostEqual(res_sep["p_two_sided"], 2.0 / math.comb(8, 4), places=12)

        # Identical distributions of size 3: compare against brute-force partitions.
        x = [1.0, 3.0, 5.0]
        y = [2.0, 4.0, 6.0]
        res_mid = exact_mann_whitney_two_sided(x, y)
        merged = x + y
        obs_u = 3.0  # (1<2,4,6)->0; (3<4,6)->1; (5<6)->2 => U_x = 0+1+2 = 3? Wait:
        # 1 beats 0 of y; 3 beats 1 of y (2); 5 beats 2 of y (2,4) => U_x = 3, U_y = 6
        extreme = 0
        total = 0
        for idx in itertools.combinations(range(6), 3):
            total += 1
            xs = [merged[i] for i in idx]
            ys = [merged[i] for i in range(6) if i not in idx]
            u = sum(1 for a in xs for b in ys if a > b)
            if u <= min(obs_u, 9 - obs_u) or u >= max(obs_u, 9 - obs_u):
                extreme += 1
        self.assertAlmostEqual(res_mid["p_two_sided"], extreme / total, places=12)

        # Tied groups: verify against brute-force enumeration over permutations.
        xt = [1.0, 2.0, 2.0, 4.0]
        yt = [2.0, 3.0, 4.0, 5.0]
        res_tied = exact_mann_whitney_two_sided(xt, yt)
        all_vals = xt + yt

        def u_stat(a_vals: list[float], b_vals: list[float]) -> float:
            s = 0.0
            for a in a_vals:
                for b in b_vals:
                    if a > b:
                        s += 1.0
                    elif a == b:
                        s += 0.5
            return s

        u_obs = u_stat(xt, yt)
        u_sym = len(xt) * len(yt) - u_obs
        lo = min(u_obs, u_sym)
        hi = max(u_obs, u_sym)
        tied_ext = 0
        tied_tot = 0
        for idx in itertools.combinations(range(8), 4):
            tied_tot += 1
            xs = [all_vals[i] for i in idx]
            ys = [all_vals[i] for i in range(8) if i not in idx]
            u = u_stat(xs, ys)
            if u <= lo + 1e-12 or u >= hi - 1e-12:
                tied_ext += 1
        self.assertAlmostEqual(res_tied["p_two_sided"], tied_ext / tied_tot, places=12)

    def test_verdict_classification_and_bonferroni(self) -> None:
        self.assertEqual(
            classify_cell_verdict(
                ratio_ci_low=1.08,
                ratio_ci_high=1.12,
                aa_band_low=0.98,
                aa_band_high=1.02,
                bonferroni_p=0.002,
            ),
            "regression",
        )
        self.assertEqual(
            classify_cell_verdict(
                ratio_ci_low=0.80,
                ratio_ci_high=0.85,
                aa_band_low=0.98,
                aa_band_high=1.02,
                bonferroni_p=0.002,
            ),
            "win",
        )
        # Fails Bonferroni threshold -> neutral
        self.assertEqual(
            classify_cell_verdict(
                ratio_ci_low=1.08,
                ratio_ci_high=1.12,
                aa_band_low=0.98,
                aa_band_high=1.02,
                bonferroni_p=0.02,
            ),
            "neutral",
        )
        # Overlaps A/A band -> neutral
        self.assertEqual(
            classify_cell_verdict(
                ratio_ci_low=1.01,
                ratio_ci_high=1.05,
                aa_band_low=0.98,
                aa_band_high=1.02,
                bonferroni_p=0.001,
            ),
            "neutral",
        )
        self.assertEqual(aggregate_pr_verdict(["neutral", "win"]), "win")
        self.assertEqual(aggregate_pr_verdict(["win", "regression"]), "regression")
        self.assertEqual(aggregate_pr_verdict(["neutral", "neutral"]), "neutral")

    def test_bootstrap_and_symmetrized_aa_band(self) -> None:
        ctrl = [100.0 + (i % 3) for i in range(14)]
        aa = [100.2 + (i % 3) for i in range(14)]
        head = [120.0 + (i % 3) for i in range(14)]
        r = bootstrap_ratio_ci(head, ctrl, resamples=500, seed=42)
        aa_band = symmetrized_aa_band(aa, ctrl, resamples=500, seed=43)
        self.assertGreater(r["ci_low"], aa_band["band_high"])
        self.assertLessEqual(aa_band["band_low"], 1.0)
        self.assertGreaterEqual(aa_band["band_high"], 1.0)


class TestPinningAndResume(unittest.TestCase):
    def test_pinning_env_single_and_multi_thread(self) -> None:
        cpus = [0, 1, 2, 3, 4, 5, 6, 7]
        c1, env1 = build_pinning_env(1, cpus, base_env={"PATH": "/usr/bin"})
        self.assertEqual(c1, [0])
        for var in SINGLE_THREAD_ENV_VARS:
            self.assertEqual(env1[var], "1")
        self.assertEqual(env1["OMP_PROC_BIND"], "close")
        self.assertEqual(env1["OMP_PLACES"], "{0}")

        c4, env4 = build_pinning_env(4, cpus, base_env={"PATH": "/usr/bin"})
        self.assertEqual(c4, [0, 1, 2, 3])
        self.assertEqual(env4["OMP_NUM_THREADS"], "4")
        self.assertEqual(env4["OMP_PROC_BIND"], "close")
        self.assertEqual(env4["GOMP_CPU_AFFINITY"], "0,1,2,3")

    def test_append_only_csv_resume_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "raw.csv"
            sample_row = {
                "pr": "pr287",
                "cell_id": "lj_n256_t1",
                "cell_kind": "comparative",
                "potential": "lj",
                "atoms": 256,
                "threads": 1,
                "round_idx": 0,
                "slot_idx": 0,
                "arm": "control",
                "commit_sha": "a" * 40,
                "tree_sha": "b" * 40,
                "total_ns_per_eval": "1234.5",
                "evals": 20,
                "omp_max_threads": 1,
                "omp_parallel_threads": 1,
                "affinity": "0",
                "omp_parallel_affinity": "0",
                "energy": "-100.0",
                "energy_bits": "0x1",
                "force_hash": "0x2",
                "list_reused": "true",
                "repeat_energy_bit_identical": "true",
                "repeat_force_bit_identical": "true",
                "repeat_max_abs_force_diff": "0",
                "small_cell_regime": "false",
            }
            append_raw_row(csv_path, sample_row)
            rows = load_raw_csv(csv_path)
            self.assertEqual(len(rows), 1)
            seen = {row_key(r) for r in rows}
            self.assertIn(("pr287", "lj_n256_t1", 0, 0, "control"), seen)

    def test_render_markdown_smoke(self) -> None:
        cell_row = {
            "cell_id": "lj_n256_t1",
            "potential": "lj",
            "atoms": 256,
            "threads": 1,
            "n_rounds": 14,
            "pilot_rel_se": 0.01,
            "achieved_rel_se": 0.01,
            "control_median_ns": 1000.0,
            "control_iqr_ns": 10.0,
            "control_aa_median_ns": 1001.0,
            "control_aa_iqr_ns": 11.0,
            "head_median_ns": 999.0,
            "head_iqr_ns": 12.0,
            "ratio": 0.999,
            "ratio_ci_low": 0.99,
            "ratio_ci_high": 1.01,
            "aa_ratio": 1.001,
            "aa_band_low": 0.99,
            "aa_band_high": 1.01,
            "aa_per_round_min": 0.98,
            "aa_per_round_max": 1.02,
            "mann_whitney_u": 95.0,
            "exact_mann_whitney_p": 0.9,
            "bonferroni_p": 1.0,
            "verdict": "neutral",
        }
        arm_entry = {
            "commit_sha": "a" * 40,
            "tree_sha": "b" * 40,
            "flag_audit": {"has_march_native": False, "has_ffast_math": False},
        }
        summary = {
            "base_develop_sha": "4aa93511d35c1cc75a93b0b372eb80c55f1365d8",
            "machine": {
                "compiler_version": "g++ 13.3.0",
                "build_type": "release",
                "cpu_model": "x86_64",
                "physical_cores": 8,
            },
            "arms": {
                "pr287": {arm: arm_entry for arm in ARMS},
                "pr293": {arm: arm_entry for arm in ARMS},
            },
            "prs": {
                "pr287": {
                    "pr_number": 287,
                    "title": "multi-image",
                    "n_comparative_cells": 1,
                    "overall_verdict": "neutral",
                    "counts": {"win": 0, "regression": 0, "neutral": 1},
                    "comparative_cells": [cell_row],
                    "small_head_only_cells": [],
                },
                "pr293": {
                    "pr_number": 293,
                    "title": "sutton-chen",
                    "n_comparative_cells": 1,
                    "overall_verdict": "neutral",
                    "counts": {"win": 0, "regression": 0, "neutral": 1},
                    "comparative_cells": [cell_row],
                    "small_head_only_cells": [],
                },
            },
            "small_cell_checks": [],
            "pr293_agreements": [],
            "pr293_term_limit_checks": [],
            "slot_arm_counts": {},
        }
        md = render_markdown(summary)
        self.assertIn("# Pre-landing Performance Report: PR #287 and PR #293", md)
        self.assertIn("**neutral**", md)


class TestHygiene(unittest.TestCase):
    def test_committed_files_lf_and_clean(self) -> None:
        bench_dir = Path(__file__).resolve().parent
        node = platform.node()
        for rel in (
            "rgpot_perf_worker.cc",
            "run_prelanding_perf.py",
            "test_prelanding_perf.py",
        ):
            raw = (bench_dir / rel).read_bytes()
            self.assertNotIn(b"\r\n", raw, f"{rel} must use LF only")
            text = raw.decode("utf-8")
            if node and len(node) >= 3:
                self.assertNotIn(node, text, f"{rel} leaked hostname")
            for name in ("home", "agent", "mnt", "opt"):
                token = "/" + name + "/"
                self.assertNotIn(token, text, f"{rel} contains machine path {token}")


if __name__ == "__main__":
    unittest.main()
