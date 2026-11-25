"""CI collects and runs the xTB regression tests.

The SCC rescue tests and the store-backed xTB cache cases once sat where
no CI job reached them: the first outside every pytest path, the second
compiled only with the cache, which the xTB job turns off.
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


def test_an_xtb_job_runs_the_cache_cases_with_the_cache_on():
    legs = [
        job
        for job in _jobs("xtb.yml").values()
        if "-Dwith_xtb=true" in job and "-Dwith_cache=true" in job
    ]
    assert len(legs) == 1
    leg = legs[0]
    assert "XTBtest EngineTableTest" in leg
    assert "--test-args '[cache-db]'" in leg
    env = re.search(r"environments: (\S+)", leg).group(1)
    pixi = tomllib.loads((ROOT / "pixi.toml").read_text(encoding="utf-8"))
    assert {"xtb", "cache"} <= set(pixi["environments"][env]["features"])


def test_the_cache_cases_carry_the_tag_the_job_selects():
    tests = ROOT / "CppCore" / "tests"
    xtb = (tests / "XTBPotTest.cc").read_text(encoding="utf-8")
    case = xtb.index('"[xtb][linked][scc][cache][cache-db]"')
    assert xtb.rindex("#ifdef RGPOT_HAS_CACHE", 0, case) > xtb.rindex("#endif", 0, case)
    engine = (tests / "EngineTableTest.cc").read_text(encoding="utf-8")
    assert '"[engine][xtb][scc][cache][cache-db]"' in engine


def test_an_xtb_job_runs_the_scc_rescue_tests_with_an_engine():
    legs = [
        job
        for job in _jobs("xtb.yml").values()
        if "tests/python/test_xtb_scc_rescue.py" in job
    ]
    assert len(legs) == 1
    leg = legs[0]
    assert "-Dwith_xtb=true" in leg
    assert "-Dwith_python=true" in leg
    assert re.search(r"export RGPOT_XTB_ENGINE=\S*libxtb_engine", leg)
    assert "python3 -c 'import rgpot'" in leg
