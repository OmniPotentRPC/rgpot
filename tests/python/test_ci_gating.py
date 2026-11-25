"""CI collects and runs the xTB regression tests.

The SCC rescue tests once sat outside every pytest path, so no CI job
collected them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Python 3.10 has no tomllib, and the wheel jobs still test on it.
tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def _jobs(workflow: str) -> dict[str, str]:
    text = (WORKFLOWS / workflow).read_text(encoding="utf-8")
    body = text.split("\njobs:\n", 1)[1]
    parts = re.split(r"^  ([A-Za-z0-9_-]+):\n", body, flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2]))


def test_scc_rescue_tests_sit_under_the_pytest_paths():
    options = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    testpaths = options["tool"]["pytest"]["ini_options"]["testpaths"]
    found = [
        path
        for name in testpaths
        for path in (ROOT / name).rglob("test_xtb_scc_rescue.py")
    ]
    assert len(found) == 1
    assert found[0].parent == ROOT / "tests" / "python"


def test_py_ci_runs_the_directory_of_the_scc_rescue_tests():
    job = _jobs("py-ci.yml")["rgpot-pytest"]
    assert re.search(r"-m pytest -q tests/python\b", job)
