"""The wheel-content checker rejects generator sources and sympy."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from check_wheel_contents import violations  # noqa: E402

CHECKER = Path(__file__).parent / "check_wheel_contents.py"
METADATA = "rgpot-1.0.dist-info/METADATA"
CLEAN = {
    "rgpot/__init__.py": b"from ._core import *\n",
    "rgpot/_core.abi3.so": b"\x7fELF",
    METADATA: b"Metadata-Version: 2.1\nName: rgpot\nRequires-Dist: numpy\n",
}


def make_wheel(tmp_path: Path, members: dict[str, bytes]) -> str:
    wheel = tmp_path / "rgpot-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return str(wheel)


def test_clean_wheel_passes(tmp_path):
    assert violations(make_wheel(tmp_path, CLEAN)) == []


@pytest.mark.parametrize("name", ["catalog.py", "cbackend.py"])
def test_generator_file_is_rejected(tmp_path, name):
    members = {**CLEAN, f"rgpot/xckernel/{name}": b"x = 1\n"}
    found = violations(make_wheel(tmp_path, members))
    assert len(found) == 1 and name in found[0], found


def test_sympy_requirement_is_rejected(tmp_path):
    members = {**CLEAN, METADATA: CLEAN[METADATA] + b"Requires-Dist: sympy>=1.12\n"}
    found = violations(make_wheel(tmp_path, members))
    assert len(found) == 1 and "sympy" in found[0], found


def test_sympy_import_is_rejected(tmp_path):
    members = {**CLEAN, "rgpot/gen.py": b"import os\nfrom sympy import symbols\n"}
    found = violations(make_wheel(tmp_path, members))
    assert found == ["module imports sympy: rgpot/gen.py"], found


def test_vendored_sympy_package_is_rejected(tmp_path):
    members = {**CLEAN, "rgpot/_vendor/sympy/core.py": b"y = 2\n"}
    found = violations(make_wheel(tmp_path, members))
    assert any("sympy package" in m for m in found), found


def test_cli_exit_status(tmp_path):
    good = make_wheel(tmp_path, CLEAN)
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad = make_wheel(bad_dir, {**CLEAN, "rgpot/catalog.py": b""})
    run = lambda *w: subprocess.run(  # noqa: E731
        [sys.executable, str(CHECKER), *w], capture_output=True
    ).returncode
    assert run(good) == 0
    assert run(bad) == 1
    assert run() == 2
