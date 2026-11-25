#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Write the head, tree, dirty flag and compiler the long-range bench logs.

    lr_build_info.py OUTPUT SOURCE_ROOT COMPILER

The build runs this every time and the file is rewritten only when a
value changes, so a commit after configure still reaches the log and an
unchanged checkout rebuilds nothing. A source tree without git leaves
the head, tree and dirty flag to the bench defaults.
"""

import subprocess
import sys
from pathlib import Path


def git(root, *args):
    return subprocess.run(
        ["git", "-C", root, *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: lr_build_info.py OUTPUT SOURCE_ROOT COMPILER")
    out, root, compiler = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    lines = ["#pragma once"]
    try:
        head = git(root, "rev-parse", "HEAD")
        tree = git(root, "rev-parse", "HEAD^{tree}")
        dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        lines += [
            f'#define RGPOT_GIT_HEAD "{head}"',
            f'#define RGPOT_GIT_TREE "{tree}"',
            f"#define RGPOT_GIT_DIRTY {1 if dirty else 0}",
        ]
    lines.append(f'#define RGPOT_COMPILER "{compiler}"')
    text = "\n".join(lines) + "\n"
    if not out.exists() or out.read_text(encoding="ascii") != text:
        out.write_text(text, encoding="ascii")


if __name__ == "__main__":
    main()
