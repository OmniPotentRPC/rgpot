# Copyright 2023--present rgpot developers
"""Release-notes fragments: the check script and the notes template."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "ci" / "release-notes.md"

_spec = importlib.util.spec_from_file_location(
    "check_news_fragments", ROOT / "scripts" / "check_news_fragments.py"
)
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

HEADER = "<!-- Copyright 2023--present rgpot developers -->\n\n"


def test_repository_fragments_are_clean():
    for path in check.fragments():
        found = check.problems(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name}: {found}"


@pytest.mark.parametrize(
    "text, expected",
    [
        (HEADER + "Fix the thing.\n", "Fix the thing.\n"),
        ("- Fix the thing.\n  More detail.\n", "Fix the thing.\nMore detail.\n"),
        ("- - Fix it.\n", "Fix it.\n"),
        ("Copyright 2023 someone\nFix it.\n", "Fix it.\n"),
        ("SPDX-License-Identifier: MIT\n\nFix it.\n", "Fix it.\n"),
        ("<!--\nmulti\nline\n-->\nFix it.\n", "Fix it.\n"),
        ("Fix it.\n\n  - a\n  - b\n", "Fix it.\n\n  - a\n  - b\n"),
        ("<!-- only a comment -->\n", ""),
    ],
)
def test_normalize(text, expected):
    assert check.normalize(text) == expected


def test_problems_flag_both_rules():
    assert check.problems(HEADER + "Fix it.\n")
    assert check.problems("- Fix it.\n")
    assert not check.problems("Fix it.\n")


def test_fix_rewrites_fragments(tmp_path):
    (tmp_path / "+a.fixed.md").write_text(HEADER + "- Fix it.\n")
    assert check.main(["--dir", str(tmp_path)]) == 1
    assert check.main(["--dir", str(tmp_path), "--fix"]) == 0
    assert (tmp_path / "+a.fixed.md").read_text() == "Fix it.\n"
    assert check.main(["--dir", str(tmp_path)]) == 0


def test_template_strips_headers_and_bullets(tmp_path):
    pytest.importorskip("towncrier")
    frag = tmp_path / "news"
    frag.mkdir()
    (frag / "+a.fixed.md").write_text(HEADER + "Plain entry.\n")
    (frag / "+b.fixed.md").write_text("- Bullet entry\n  continued here.\n")
    (frag / "+c.added.md").write_text(
        "<!--\nblock\n-->\nCopyright 2023 x\n\n- Added entry.\n"
    )
    (frag / "+d.fixed.md").write_text("Copyright 2023 x\nPlain two.\n")
    config = tmp_path / "towncrier.toml"
    config.write_text(
        "[tool.towncrier]\n"
        'name = "demo"\n'
        'version = "1.0.0"\n'
        f'directory = "{frag.as_posix()}"\n'
        f'template = "{TEMPLATE.as_posix()}"\n'
        'filename = "CHANGELOG.md"\n'
        'title_format = "## [{version}]"\n'
        'underlines = ["", "", ""]\n'
        "[[tool.towncrier.type]]\n"
        'directory = "added"\nname = "Added"\nshowcontent = true\n'
        "[[tool.towncrier.type]]\n"
        'directory = "fixed"\nname = "Fixed"\nshowcontent = true\n'
    )
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "towncrier",
            "build",
            "--draft",
            "--config",
            str(config),
            "--dir",
            str(tmp_path),
            "--version",
            "1.0.0",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "Copyright" not in out
    assert "<!--" not in out
    assert "block" not in out
    assert "- - " not in out
    assert "- Plain entry." in out
    assert "- Plain two." in out
    assert "- Bullet entry\n  continued here." in out
    assert "- Added entry." in out
