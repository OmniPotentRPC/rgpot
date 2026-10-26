#!/usr/bin/env python3
"""Append the human-readable ref as a trailing comment on SHA-pinned `uses:` lines.

YAML exported from Nickel carries no comments, so the version that each pinned
SHA stands for is read from the trailing comments in ci/gha/lib/pins.ncl.

Usage: annotate-pins.py WORKFLOW.yml
"""

import re
import sys
from pathlib import Path

PINS = Path(__file__).parent / "lib" / "pins.ncl"
PIN_LINE = re.compile(r'"([\w.-]+/[\w./-]+@[0-9a-f]{40})",\s*#\s*(\S+)')
USES_LINE = re.compile(r"^(\s*(?:-\s*)?uses:\s*)([\w.-]+/[\w./-]+@[0-9a-f]{40})\s*$")


def main(path: str) -> None:
    refs = dict(PIN_LINE.findall(PINS.read_text()))
    out = []
    for line in Path(path).read_text().split("\n"):
        m = USES_LINE.match(line)
        if m:
            if m[2] not in refs:
                sys.exit(f"{path}: pinned action {m[2]} has no version comment in pins.ncl")
            line = f"{m[1]}{m[2]} # {refs[m[2]]}"
        out.append(line)
    Path(path).write_text("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
