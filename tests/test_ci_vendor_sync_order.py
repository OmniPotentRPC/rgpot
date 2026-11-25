# Copyright 2023--present rgpot developers
"""The CppCore vendor sync check runs cargo, so it needs the pinned env."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci-orchestrator.yml"
SYNC = "scripts/check_rgpot_core_cppcore_sync.sh"


def _steps_of_job_running(command: str) -> list[str]:
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines) if command in line)
    start = max(i for i in range(at) if lines[i].strip() == "steps:")
    indent = len(lines[start]) - len(lines[start].lstrip())
    steps: list[list[str]] = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        if re.match(r"\s*- ", line) and (
            not steps or len(line) - len(line.lstrip()) <= steps[0][0].index("-")
        ):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    return ["\n".join(step) for step in steps]


def test_vendor_sync_runs_after_the_pixi_environment():
    steps = _steps_of_job_running(SYNC)
    sync = next(i for i, step in enumerate(steps) if SYNC in step)
    pixi = [i for i, step in enumerate(steps) if "prefix-dev/setup-pixi" in step]
    assert pixi and pixi[0] < sync, steps[sync]
    assert "shell: pixi run bash" in steps[sync], steps[sync]
