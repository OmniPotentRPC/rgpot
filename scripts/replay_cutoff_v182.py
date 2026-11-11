#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Replay prepared trees with commit-tree.

No author or committer date is set. The clock assigns both,
so a commit may share its parent's second and is never given
an earlier one. Archive refs are not pushed. The caller
updates refs after the log is written.
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: replay_cutoff_v182.py parent manifest", file=sys.stderr)
        return 2
    for key in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):
        os.environ.pop(key, None)
    os.environ["GIT_AUTHOR_NAME"] = "Rohit Goswami"
    os.environ["GIT_AUTHOR_EMAIL"] = "rgoswami@ieee.org"
    os.environ["GIT_COMMITTER_NAME"] = "Rohit Goswami"
    os.environ["GIT_COMMITTER_EMAIL"] = "rgoswami@ieee.org"
    os.environ["TZ"] = "America/Chicago"
    parent = sys.argv[1]
    for line in open(sys.argv[2]):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tree, message = line.split(" ", 1)
        new = subprocess.check_output(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "commit-tree",
                tree,
                "-p",
                parent,
                "-F",
                message,
            ],
            text=True,
        ).strip()
        info = subprocess.check_output(
            ["git", "log", "-1", "--format=%aI %cI %G? %H", new],
            text=True,
        ).strip()
        print(f"{parent} {new} {info}")
        parent = new
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
