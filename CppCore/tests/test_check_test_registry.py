#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Regression test for intermixed CLI parsing in check_test_registry.py."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "CppCore" / "tests" / "check_test_registry.py"
REGISTRY = ROOT / "CppCore" / "tests" / "registry.txt"

sys.path.insert(0, str(SCRIPT.parent))
import check_test_registry as ctr  # noqa: E402


class CheckTestRegistryIntermixedArgsTests(unittest.TestCase):
    def test_lint_accepts_options_and_rejects_positional_features(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--ctest",
                "custom-ctest",
                "lint",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("entries", proc.stdout)

        bad = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "lint",
                "tests=1",
                "--ctest",
                "custom-ctest",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(bad.returncode, 2)

    def test_cmake_mode_accepts_intermixed_ctest_and_list_flags(self) -> None:
        entries, errors = ctr.parse(REGISTRY)
        self.assertEqual(errors, [])

        all_atoms = sorted(
            {atom for e in entries.values() for clause in e.gate for atom in clause}
            - {ctr.PROBE}
        )
        active = {"tests", "python"}
        features = {atom: (atom in active) for atom in all_atoms}

        expected = [
            e
            for e in entries.values()
            if e.reason is None and e.holds(features, probe=False)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            build_dir = tmpdir / "build"
            build_dir.mkdir()
            list_out = tmpdir / "list.txt"

            ctest_tests = []
            for entry in expected:
                workdir = (
                    str(build_dir)
                    if entry.workdir is None
                    else str(ROOT if entry.workdir == "." else ROOT / entry.workdir)
                )
                props = [
                    {"name": "TIMEOUT", "value": float(entry.timeout)},
                    {"name": "WORKING_DIRECTORY", "value": workdir},
                ]
                if entry.suite:
                    props.append({"name": "LABELS", "value": [entry.suite]})
                if entry.env:
                    props.append(
                        {
                            "name": "ENVIRONMENT",
                            "value": [f"{k}={v}" for k, v in entry.env.items()],
                        }
                    )
                if entry.serial:
                    props.append({"name": "RUN_SERIAL", "value": True})
                if entry.should_fail:
                    props.append({"name": "WILL_FAIL", "value": True})
                test = {"name": entry.name, "backtrace": 0, "properties": props}
                if entry.kind != "custom":
                    exe, tail = ctr.expected_arguments(entry, entries)
                    program = sys.executable if exe is None else str(build_dir / exe)
                    test["command"] = [program, *tail]
                ctest_tests.append(test)

            payload = {
                "backtraceGraph": {
                    "files": [str(ROOT / "CppCore" / "tests" / "tests.cmake")],
                    "nodes": [{"file": 0}],
                },
                "tests": ctest_tests,
            }
            payload_file = tmpdir / "ctest.json"
            payload_file.write_text(json.dumps(payload), encoding="utf-8")

            fake_ctest = tmpdir / "fake_ctest.py"
            fake_ctest.write_text(
                f"#!{sys.executable}\n"
                "import pathlib, sys\n"
                "assert '--show-only=json-v1' in sys.argv\n"
                f"sys.stdout.write(pathlib.Path({str(payload_file)!r}).read_text())\n",
                encoding="utf-8",
            )
            fake_ctest.chmod(fake_ctest.stat().st_mode | stat.S_IXUSR)

            feature_args = [f"{k}={int(v)}" for k, v in sorted(features.items())]
            argv = [
                sys.executable,
                str(SCRIPT),
                "cmake",
                str(build_dir),
                feature_args[0],
                "--ctest",
                str(fake_ctest),
                feature_args[1],
                "--list",
                str(list_out),
                *feature_args[2:],
            ]
            proc = subprocess.run(argv, capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            written = list_out.read_text(encoding="utf-8")
            self.assertIn("test test_list_parity\n", written)
            self.assertIn("test check_test_registry_selftest\n", written)


class RerunGateTests(unittest.TestCase):
    def test_rerun_must_carry_its_target_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "registry.txt"
            registry.write_text(
                "A | tests core thread_registry | unit a ThreadInvariantTest.cc |\n"
                "A_again | tests core | rerun A |\n",
                encoding="utf-8",
            )
            _, errors = ctr.parse(registry)
        self.assertEqual(len(errors), 1, msg=errors)
        self.assertIn("rerun gate lacks 'thread_registry' from A", errors[0])

    def test_windows_shared_drops_both_thread_invariant_entries(self) -> None:
        entries, errors = ctr.parse(REGISTRY)
        self.assertEqual(errors, [])
        atoms = {a for e in entries.values() for c in e.gate for a in c} - {ctr.PROBE}
        names = ("ThreadInvariantTest", "ThreadInvariantTest_limit1")
        on = {atom: True for atom in atoms}
        windows_shared = dict(on, thread_registry=False, posix=False, linux=False)
        for name in names:
            self.assertTrue(entries[name].holds(on, probe=False), name)
            self.assertFalse(entries[name].holds(windows_shared, probe=False), name)
        self.assertTrue(entries["ThreadReachTest"].holds(windows_shared, probe=False))

    def test_both_builds_derive_the_atom_from_the_library_type(self) -> None:
        meson = (ROOT / "CppCore" / "meson.build").read_text(encoding="utf-8")
        cmake = (ROOT / "CppCore" / "tests" / "tests.cmake").read_text(encoding="utf-8")
        self.assertRegex(
            meson,
            r"'thread_registry': not \(\s*host_machine\.system\(\) == 'windows'"
            r"\s*and get_option\('default_library'\) != 'static'\s*\)",
        )
        self.assertIn(
            "rgpot_test_set_feature(thread_registry NOT (WIN32 AND BUILD_SHARED_LIBS))",
            cmake,
        )
        self.assertNotRegex(meson, r"_rt\[0\] == 'ThreadInvariantTest'\s*and host_machine")


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
