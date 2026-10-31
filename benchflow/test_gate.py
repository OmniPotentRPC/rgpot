"""Proofs for the calibration gate.

Run from the repository root:

    python -m unittest benchflow.test_gate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchflow.export_results import (  # noqa: E402
    CANONICAL,
    MACHINE_NAME,
    check_machine,
    command_gate,
    decide_status,
    derive_conf,
    format_gate_lines,
    gate_rows,
    prepare_machine,
    relative_threshold,
    select_result,
)


def _result(folder: Path, machine: str, commit: str, env: str, params: dict | None = None, name: str | None = None):
    directory = folder / machine
    directory.mkdir(parents=True, exist_ok=True)
    record = {"arch": "x86_64", "cpu": "unspecified", "machine": machine, "os": "Linux", "ram": "unspecified"}
    if params:
        record.update(params)
    payload = {
        "commit_hash": commit,
        "env_name": env,
        "params": record,
        "version": 2,
    }
    filename = name or f"{commit[:8]}-{env}.json"
    path = directory / filename
    path.write_text(json.dumps(payload) + "\n")
    return path


class TestRegression(unittest.TestCase):
    def test_regression_fails_and_clean_ratio_passes(self):
        thresholds = [{"factor": 1.05, "kind": "time", "name": "bench.time_work", "params": [16]}]
        slow = gate_rows(
            [{"name": "bench.time_work", "params": [16], "ratio": 1.20}],
            thresholds,
        )
        self.assertEqual(len(slow), 1)
        self.assertEqual(slow[0]["reason"], "regression")
        self.assertEqual(decide_status("a" * 40, "b" * 40, slow, True), 1)
        clean = gate_rows(
            [{"name": "bench.time_work", "params": [16], "ratio": 1.01}],
            thresholds,
        )
        self.assertEqual(clean, [])
        self.assertEqual(decide_status("a" * 40, "b" * 40, clean, True), 0)

    def test_equal_revisions_fail_without_a_comparison(self):
        commit = "a" * 40
        self.assertEqual(decide_status(commit, commit, [], True), 1)
        self.assertEqual(decide_status(commit, commit, None, False), 1)

    def test_missing_result_fails_even_when_asv_exited_zero(self):
        self.assertEqual(decide_status("a" * 40, "b" * 40, None, False), 1)


class TestMissingThreshold(unittest.TestCase):
    def test_a_timing_row_without_a_threshold_fails_the_gate(self):
        measurements = [
            {"name": "bench.time_known", "params": [], "ratio": 1.01},
            {"name": "bench.time_extra", "params": [], "ratio": 1.01},
        ]
        thresholds = [
            {"name": "bench.time_known", "params": [], "factor": 1.05, "kind": "time"}
        ]
        failures = gate_rows(measurements, thresholds)
        self.assertEqual([row["reason"] for row in failures], ["no-threshold"])
        self.assertEqual(failures[0]["name"], "bench.time_extra")
        lines = format_gate_lines(measurements, thresholds, failures)
        extra = next(line for line in lines if line.startswith("bench.time_extra"))
        self.assertTrue(extra.endswith("fail"))
        self.assertEqual(decide_status("a" * 40, "b" * 40, failures, True), 1)


class TestAsvStatus(unittest.TestCase):
    def test_nonzero_asv_status_is_written_after_the_report(self):
        base = "a" * 40
        head = "b" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            self._write_result(results, base, [1.0, 1.0, 1.0])
            self._write_result(results, head, [1.0, 1.0, 1.0])
            thresholds = root / "thresholds.json"
            thresholds.write_text(
                json.dumps(
                    {
                        "benchmarks": [
                            {
                                "factor": 2.0,
                                "kind": "time",
                                "name": "bench.time_work",
                                "params": [],
                            }
                        ]
                    }
                )
            )
            report = root / "report.txt"
            status = root / "status.txt"
            command_gate(
                argparse.Namespace(
                    base=base,
                    head=head,
                    machine=MACHINE_NAME,
                    results=str(results),
                    thresholds=str(thresholds),
                    asv_log=None,
                    asv_status=2,
                    report=str(report),
                    status=str(status),
                )
            )
            self.assertTrue(report.is_file())
            self.assertIn("asv_status=2", report.read_text())
            self.assertEqual(status.read_text(), "2\n")

    def _write_result(self, results: Path, commit: str, samples: list[float]) -> None:
        folder = results / MACHINE_NAME
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "commit_hash": commit,
            "date": 1,
            "env_name": "virtualenv-py3.12",
            "env_vars": {},
            "params": dict(CANONICAL),
            "python": "3.12",
            "requirements": {},
            "result_columns": ["result", "params", "samples"],
            "results": {"bench.time_work": [[1.0], [], samples]},
            "version": 2,
        }
        (folder / f"{commit[:8]}-virtualenv-py3.12.json").write_text(json.dumps(payload))


class TestReadelf(unittest.TestCase):
    def test_missing_readelf_exits_before_the_elf_parse(self):
        from benchflow.build_wheel import repair_wheel

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "cache"
            cache.mkdir()
            wheel_root = root / "wheel"
            libs = wheel_root / "rgpot" / ".rgpot.mesonpy.libs"
            libs.mkdir(parents=True)
            (libs / "librgpot.so.1.2.3").write_text("not an elf object")
            distinfo = wheel_root / "rgpot-0.1.0.dist-info"
            distinfo.mkdir()
            (distinfo / "METADATA").write_text("Name: rgpot\n")
            (distinfo / "RECORD").write_text("")
            wheel = cache / "rgpot-0.1.0-py3-none-linux_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                for path in wheel_root.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(wheel_root).as_posix())
            before = wheel.read_bytes()
            bindir = root / "bin"
            bindir.mkdir()
            (bindir / "bash").symlink_to(shutil.which("bash"))
            old = os.environ.get("PATH")
            os.environ["PATH"] = str(bindir)
            try:
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    repair_wheel(str(cache))
            finally:
                if old is None:
                    os.environ.pop("PATH", None)
                else:
                    os.environ["PATH"] = old
            self.assertIn("--refresh-soname", caught.exception.cmd)
            self.assertEqual(wheel.read_bytes(), before)


class TestPerRowGate(unittest.TestCase):
    def test_each_row_uses_its_own_factor(self):
        thresholds = [
            {"factor": 1.05, "kind": "time", "name": "bench.time_quiet", "params": []},
            {"factor": 1.40, "kind": "time", "name": "bench.time_noisy", "params": []},
        ]
        measurements = [
            {"name": "bench.time_quiet", "params": [], "ratio": 1.10},
            {"name": "bench.time_noisy", "params": [], "ratio": 1.10},
        ]
        failures = gate_rows(measurements, thresholds)
        self.assertEqual([row["name"] for row in failures], ["bench.time_quiet"])
        self.assertGreater(max(row["factor"] for row in thresholds), 1.10)
        self.assertNotIn("max(factors)", (ROOT / "benchflow" / "export_results.py").read_text())


class TestSelectResult(unittest.TestCase):
    def test_one_match_is_kept_and_mismatches_fail(self):
        commit = "0123456789abcdef" * 2 + "01234567"
        other = "fedcba9876543210" * 2 + "fedcba98"
        env = "virtualenv-py3.12-numpy2.5.4"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _result(root, MACHINE_NAME, commit, env)
            _result(root, MACHINE_NAME, other, env)
            _result(
                root,
                "other-machine",
                commit,
                env,
                params={"cpu": platform.node(), "machine": "other-machine"},
            )
            path, data = select_result(root, commit, MACHINE_NAME)
            self.assertEqual(data["commit_hash"], commit)
            self.assertEqual(path.parent.name, MACHINE_NAME)
            node = platform.node()
            if node:
                self.assertNotIn(node, json.dumps(data["params"]))

            with self.assertRaises(SystemExit):
                select_result(root, "c" * 40, MACHINE_NAME)

            _result(root, MACHINE_NAME, commit, env, name=f"copy-{commit[:8]}-{env}.json")
            with self.assertRaises(SystemExit):
                select_result(root, commit, MACHINE_NAME)

    def test_two_environments_fail(self):
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _result(root, MACHINE_NAME, commit, "env-a")
            _result(root, MACHINE_NAME, commit, "env-b")
            with self.assertRaises(SystemExit):
                select_result(root, commit, MACHINE_NAME)

    def test_selected_result_rejects_a_node_name(self):
        commit = "b" * 40
        node = platform.node()
        if not node:
            self.skipTest("platform.node() is empty")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _result(root, MACHINE_NAME, commit, "env-a", params={"cpu": node})
            with self.assertRaises(SystemExit) as caught:
                select_result(root, commit, MACHINE_NAME)
            self.assertIn("node name", str(caught.exception))
            self.assertNotIn(node, str(caught.exception))


class TestMachineRecord(unittest.TestCase):
    def test_check_ignores_an_unrelated_hostname_record(self):
        node = platform.node()
        if not node or len(node) < 3:
            self.skipTest("platform.node() is too short to test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".asv-machine.json"
            path.write_text(
                json.dumps(
                    {
                        "other": {"cpu": node, "machine": "other", "os": node},
                        "version": 1,
                        MACHINE_NAME: dict(CANONICAL),
                    }
                )
            )
            check_machine(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved[MACHINE_NAME], CANONICAL)
            self.assertIn(node, json.dumps(saved["other"]))
            self.assertNotIn(node, json.dumps(saved[MACHINE_NAME]))
            saved[MACHINE_NAME]["cpu"] = node
            path.write_text(json.dumps(saved))
            with self.assertRaises(SystemExit):
                # check_machine rewrites the selected record, so contaminate
                # after a direct assertion instead.
                from benchflow.export_results import assert_selected_record

                assert_selected_record(saved[MACHINE_NAME], MACHINE_NAME)

    def test_asv_machine_command_then_check_keeps_only_the_selected_record_clean(self):
        if shutil.which("asv") is None:
            self.skipTest("asv is not on PATH")
        node = platform.node()
        if not node or len(node) < 3:
            self.skipTest("platform.node() is too short to test")
        snakefile = (ROOT / "Snakefile").read_text()
        for flag in ("--os Linux", "--arch x86_64", "--cpu unspecified", "--num_cpu 1", "--ram unspecified"):
            self.assertIn(flag, snakefile)
        env = os.environ.copy()
        cargo = env.get("CARGO_HOME")
        rustup = env.get("RUSTUP_HOME")
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()
            if not cargo:
                env["CARGO_HOME"] = str(Path(env["HOME"]) / ".cargo")
            if not rustup:
                env["RUSTUP_HOME"] = str(Path(env["HOME"]) / ".rustup")
            env["HOME"] = str(home)
            path = home / ".asv-machine.json"
            path.write_text(
                json.dumps(
                    {
                        "other": {"cpu": node, "machine": "other"},
                        "version": 1,
                    }
                )
            )
            prepare_machine(path)
            completed = subprocess.run(
                [
                    "asv",
                    "machine",
                    "--yes",
                    "--machine",
                    MACHINE_NAME,
                    "--os",
                    "Linux",
                    "--arch",
                    "x86_64",
                    "--cpu",
                    "unspecified",
                    "--num_cpu",
                    "1",
                    "--ram",
                    "unspecified",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            check_machine(path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved[MACHINE_NAME], CANONICAL)
            self.assertIn(node, json.dumps(saved["other"]))
            self.assertNotIn(node, json.dumps(saved[MACHINE_NAME]))


class TestThresholdCovers(unittest.TestCase):
    def test_gate_covers_every_calibration_ratio(self):
        ratios = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.015, 0.99, 1.005, 1.0646]
        stats = relative_threshold(ratios)
        worst = max(abs(ratio - 1.0) for ratio in ratios)
        self.assertGreaterEqual(stats["relative_threshold"], worst)
        self.assertAlmostEqual(worst, 0.0646, places=4)
        self.assertGreaterEqual(stats["relative_threshold"], 0.05)
        self.assertGreaterEqual(stats["relative_threshold"], stats["three_mad"])
        self.assertGreaterEqual(stats["relative_threshold"], stats["p99_abs_ratio_minus_1"])
        factor = stats["factor"]
        for ratio in ratios:
            self.assertLessEqual(ratio, factor)
            self.assertLessEqual(abs(ratio - 1.0), stats["relative_threshold"])

    def test_floor_still_covers_a_quiet_distribution(self):
        ratios = [1.001, 0.999, 1.002, 0.998, 1.0, 1.001, 0.999, 1.0, 1.002, 0.997]
        stats = relative_threshold(ratios)
        self.assertEqual(stats["relative_threshold"], 0.05)
        self.assertGreaterEqual(stats["relative_threshold"], stats["max_abs_ratio_minus_1"])


class TestPins(unittest.TestCase):
    def test_requirements_are_exact_and_used_by_the_asv_config(self):
        requirements = (ROOT / "benchflow" / "requirements.txt").read_text().splitlines()
        config = json.loads((ROOT / "asv.conf.json").read_text())
        config_text = json.dumps(config)
        host_only = {"snakemake", "asv", "virtualenv", "packaging"}
        matrix_names = {"numpy", "asv_bench_memray", "memray"}
        pins = []
        for line in requirements:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            self.assertNotIn(">=", stripped)
            self.assertNotIn(">", stripped.split("==", 1)[-1])
            name, version = stripped.split("==")
            self.assertTrue(version)
            pins.append((name, version))
        self.assertGreaterEqual(len(pins), 8)
        build_text = "\n".join(config["build_command"])
        self.assertNotIn(">=", build_text)
        for name, version in pins:
            if name in host_only:
                continue
            if name in matrix_names:
                self.assertIn(version, config["matrix"]["req"][name])
            else:
                self.assertIn(f"{name}=={version}", build_text)
        self.assertNotIn(">=", config_text)
        wheel = ROOT / "benchflow" / "build_wheel.py"
        if wheel.is_file():
            self.assertNotIn(">=", wheel.read_text())

    def test_derive_conf_points_at_the_requested_results_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "asv.run.conf.json"
            derive_conf(ROOT / "asv.conf.json", dest, "benchflow/out/results/time-01")
            data = json.loads(dest.read_text())
            self.assertEqual(data["results_dir"], "benchflow/out/results/time-01")
            self.assertEqual(data["repo"], ".")


class TestSnakefile(unittest.TestCase):
    def test_workflow_text(self):
        text = (ROOT / "Snakefile").read_text()
        self.assertNotIn("|| true", text)
        self.assertNotIn("--quick", text)
        self.assertNotIn("--skip-existing", text)
        self.assertIn('exit "$status"', text)
        self.assertEqual(text.count("derive-conf"), 3)
        self.assertEqual(text.count("asv run "), 2)
        self.assertEqual(text.count("asv continuous "), 1)
        self.assertGreaterEqual(text.count("--interleave-rounds"), 3)
        self.assertIn("--record-samples", text)
        self.assertIn("benchflow/out/results/time-$id", text)
        self.assertIn("benchflow/out/results/pr", text)
        self.assertIn(". benchflow/env.sh", text)
        self.assertIn("bench-runner", (ROOT / "benchflow" / "config.yaml").read_text())
        node = platform.node()
        if node and len(node) >= 3:
            for relative in (
                "Snakefile",
                "asv.conf.json",
                "benchflow/config.yaml",
                "benchflow/env.sh",
                "benchflow/export_results.py",
                "benchflow/requirements.txt",
                "benchflow/README.md",
                "benchflow/test_gate.py",
            ):
                path = ROOT / relative
                if path.is_file():
                    self.assertNotIn(node, path.read_text(errors="replace"), relative)


if __name__ == "__main__":
    unittest.main()
