#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Check the release-notes fragments under docs/newsfragments.

A fragment is the text of one release-notes entry.  The release-notes
template, ci/release-notes.md, already drops what this script rejects, so
the notes stay clean either way.  The files should say the same thing:

* no copyright or SPDX header line and no comment block, because the
  fragment is the text that ends up in the notes;
* no leading list marker, because the template writes the bullet, and a
  fragment that starts with "- " would otherwise render as "- - ".

Run with no arguments to check, or with --fix to rewrite the files.
The module uses the standard library only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAGMENT_DIR = ROOT / "docs" / "newsfragments"
NOT_FRAGMENTS = {"README.md", ".gitkeep"}
HEADER_PREFIXES = ("copyright ", "(c) ", "\u00a9", "spdx-")
BULLETS = ("- ", "* ")


def is_header_line(line: str) -> bool:
    low = line.strip().lower()
    return low.startswith(HEADER_PREFIXES) or low.startswith("[//]:")


def normalize(text: str) -> str:
    """Return the fragment without comment blocks, headers or a bullet."""
    kept: list[str] = []
    in_comment = False
    for line in text.splitlines():
        low = line.strip().lower()
        if in_comment:
            in_comment = "-->" not in line
        elif low.startswith("<!--"):
            in_comment = "-->" not in low[4:]
        elif is_header_line(line):
            continue
        elif low or kept:
            kept.append(line.rstrip())
    while kept and not kept[-1]:
        kept.pop()
    for _ in range(3):
        if kept and kept[0].startswith(BULLETS):
            tail = [x[2:] if x.startswith("  ") else x for x in kept[1:]]
            kept = [kept[0][2:].lstrip()] + tail
    return "\n".join(kept) + "\n" if kept else ""


def problems(text: str) -> list[str]:
    found = []
    lines = text.splitlines()
    if any(
        x.strip().startswith("<!--") or is_header_line(x) for x in lines
    ):
        found.append("carries a copyright line or comment block")
    if text.lstrip().startswith(BULLETS):
        found.append("starts with a list marker")
    if not normalize(text).strip():
        found.append("has no text")
    return found


def fragments(directory: Path = FRAGMENT_DIR) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.name not in NOT_FRAGMENTS
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fix", action="store_true", help="rewrite files")
    parser.add_argument("--dir", type=Path, default=FRAGMENT_DIR)
    args = parser.parse_args(argv)

    bad = 0
    for path in fragments(args.dir):
        text = path.read_text(encoding="utf-8")
        found = problems(text)
        if not found:
            continue
        if args.fix and normalize(text):
            path.write_text(normalize(text), encoding="utf-8")
            print(f"fixed {path.name}: {'; '.join(found)}")
            continue
        bad += 1
        print(f"{path}: {'; '.join(found)}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
