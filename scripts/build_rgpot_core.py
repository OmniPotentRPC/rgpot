#!/usr/bin/env python3
"""Build rgpot-core for Meson and copy its static archive to @OUTPUT@."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("debug", "release"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features", default="")
    args = parser.parse_args()

    command = [
        args.cargo,
        "build",
        "--manifest-path",
        str(args.manifest),
        "--target-dir",
        str(args.target_dir),
    ]
    if args.profile == "release":
        command.append("--release")
    if args.features:
        command.extend(["--features", args.features])
    subprocess.run(command, cwd=args.manifest.parent.parent, check=True)
    source = args.target_dir / args.profile / "librgpot_core.a"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
