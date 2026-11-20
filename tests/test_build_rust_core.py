# Copyright 2023--present rgpot developers
"""Toolchain choice for the Rust core and the fast pair list.

A core-only build has no rustc on PATH and a cargo that fails. It must
reach that cargo. The 1.89 check belongs to the pair list, which still
exits with its own message when the toolchain is missing.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "scripts" / "build_rust_core.py"
FAST = ROOT / "scripts" / "build_fastnlist.py"
MESON = ROOT / "meson.build"

FAKE_CARGO = """\
import json, os, sys
argv = sys.argv[1:]
target = argv[argv.index("--target-dir") + 1]
profile = "release" if "--release" in argv else "debug"
out = os.path.join(target, profile, "librgpot_core.a")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "wb") as handle:
    handle.write(b"!<arch>\\n")
with open(os.environ["FAKE_CARGO_LOG"], "w") as handle:
    json.dump(
        {
            "argv": argv,
            "ar": os.environ.get("AR"),
            "ranlib": os.environ.get("RANLIB"),
        },
        handle,
    )
"""


def _command_slice(text: str, artifact: str) -> str:
    """Meson command list from ``--artifact`` through the first output."""
    needle = "'--artifact',\n                " + artifact + ","
    start = text.index(needle)
    return text[start : text.index("'@OUTPUT0@'", start)]


def test_core_only_reproducer_skips_toolchain_check(tmp_path: Path) -> None:
    """Core-only cargo runs without the pair list's 1.89 check.

    The fixed core-only target does not pass ``--fast-toolchain``. While
    that flag is still in the target, this is the command Meson runs, and
    it exits on the toolchain check.
    """
    extra: list[str] = []
    if "'--fast-toolchain'" in _command_slice(
        MESON.read_text(), "_rust_archive"
    ):
        extra.append("--fast-toolchain")
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    proc = subprocess.run(
        [
            "/usr/bin/python3",
            os.fspath(CORE),
            "--cargo",
            "/bin/false",
            "--manifest",
            "rgpot-core/Cargo.toml",
            "--target-dir",
            os.fspath(tmp_path / "unused"),
            "--profile",
            "debug",
            "--artifact",
            "librgpot_core.a",
            *extra,
            os.fspath(tmp_path / "core.a"),
            os.fspath(tmp_path / "core.h"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = proc.stdout + proc.stderr
    assert "rustc >= 1.89" not in combined
    assert "rgpot_fastnlist needs rustc" not in combined
    assert proc.returncode == 1


def test_fastnlist_build_enforces_189(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    proc = subprocess.run(
        [
            "/usr/bin/python3",
            os.fspath(FAST),
            "--copy",
            "/bin/false",
            "CppCore/rgpot/nlist/fastnlist/Cargo.toml",
            os.fspath(tmp_path / "target"),
            os.fspath(tmp_path / "librgpot_fastnlist.a"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert (
        "rgpot_fastnlist needs rustc >= 1.89 so the AVX-512 linkcell and "
        "minimage kernels are built. The rustc on PATH is older, and "
        "rustup has no 1.89.0 toolchain."
    ) in proc.stderr


def test_bundle_fast_toolchain_enforces_189(tmp_path: Path) -> None:
    bundle = _command_slice(MESON.read_text(), "_fast_artifact")
    assert "'--fast-toolchain'" in bundle
    env = os.environ.copy()
    env["PATH"] = "/nonexistent"
    proc = subprocess.run(
        [
            "/usr/bin/python3",
            os.fspath(CORE),
            "--cargo",
            "/bin/false",
            "--manifest",
            "rgpot-bundle/Cargo.toml",
            "--target-dir",
            os.fspath(tmp_path / "unused"),
            "--profile",
            "debug",
            "--artifact",
            "librgpot_bundle.a",
            "--fast-toolchain",
            os.fspath(tmp_path / "bundle.a"),
            os.fspath(tmp_path / "bundle.h"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "rgpot_fastnlist needs rustc >= 1.89" in proc.stderr
    assert "rustup has no 1.89.0 toolchain." in proc.stderr


@pytest.mark.skipif(
    not (
        os.access("/usr/bin/ar", os.X_OK)
        and os.access("/usr/bin/ranlib", os.X_OK)
    ),
    reason="needs /usr/bin/ar and /usr/bin/ranlib",
)
def test_core_only_uses_system_ar_on_darwin(tmp_path: Path, monkeypatch) -> None:
    cargo = tmp_path / "cargo"
    cargo.write_text(f"#!{sys.executable}\n{FAKE_CARGO}")
    cargo.chmod(cargo.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "cargo.json"
    monkeypatch.setenv("FAKE_CARGO_LOG", os.fspath(log))
    monkeypatch.setenv("AR", "arm64-apple-darwin20.0.0-ar")
    monkeypatch.setenv("RANLIB", "arm64-apple-darwin20.0.0-ranlib")
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_rust_core.py",
            "--cargo",
            os.fspath(cargo),
            "--manifest",
            "rgpot-core/Cargo.toml",
            "--target-dir",
            os.fspath(tmp_path / "target"),
            "--profile",
            "debug",
            "--artifact",
            "librgpot_core.a",
            os.fspath(tmp_path / "librgpot_core.a"),
            os.fspath(tmp_path / "rgpot_rust_built.h"),
        ],
    )
    sys.path.insert(0, os.fspath(ROOT / "scripts"))
    import build_rust_core

    build_rust_core.main()
    recorded = json.loads(log.read_text())
    assert recorded["argv"][0] == "build"
    assert recorded["ar"] == "/usr/bin/ar"
    assert recorded["ranlib"] == "/usr/bin/ranlib"
