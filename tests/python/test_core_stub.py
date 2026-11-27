# Copyright 2023--present rgpot developers
"""The committed ``rgpot/_core.pyi`` matches the bindings of this build.

The committed stub is generated from the wheel, which enables more features
than a reduced build. Every class member and module function that stubgen
emits for this build must appear in the committed stub with the same
signature and docstring; the stub may hold more, from features this build
leaves out.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

pytest.importorskip("nanobind.stubgen")

COMMITTED = Path(__file__).resolve().parents[2] / "python" / "rgpot" / "_core.pyi"


def _defs(body) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name].append(ast.dump(node))
    return out


def _members(tree: ast.Module) -> dict[str, dict[str, list[str]]]:
    members = {"": _defs(tree.body)}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members[node.name] = _defs(node.body)
    return members


def test_stubgen_of_this_build_agrees_with_the_committed_stub(tmp_path):
    generated = tmp_path / "_core.pyi"
    subprocess.run(
        [sys.executable, "-m", "nanobind.stubgen", "-m", "rgpot._core", "-o", str(generated)],
        check=True,
        capture_output=True,
    )
    live = _members(ast.parse(generated.read_text()))
    committed = _members(ast.parse(COMMITTED.read_text()))
    stale = []
    for owner, defs in live.items():
        for name, dumps in defs.items():
            if committed.get(owner, {}).get(name) != dumps:
                stale.append(f"{owner}.{name}" if owner else name)
    assert stale == [], f"rebuild python/rgpot/_core.pyi; stale: {sorted(stale)}"
