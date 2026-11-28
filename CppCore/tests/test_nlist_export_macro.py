#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""RGPOT_NLIST_API never asks a consumer for dllimport.

Neighbours.hpp marks Neighbours::nextGeneration with RGPOT_NLIST_API. A
consumer of a static librgpot on Windows has no import library, so a
dllimport there leaves __imp_ references that nothing defines. A consumer
of the shared library still resolves the call through the import library
without it. The macro block is run through the C preprocessor with and
without _WIN32 and RGPOT_NLIST_BUILDING.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER = Path("CppCore/rgpot/nlist/Neighbours.hpp")
MARK = "@@expansion@@"

CASES = (
    ("Windows consumer", ["-D_WIN32"], ""),
    ("Windows librgpot", ["-D_WIN32", "-DRGPOT_NLIST_BUILDING"], "__declspec(dllexport)"),
    ("other hosts", [], '__attribute__((visibility("default")))'),
)


def macro_block(text: str) -> str:
    lines = text.splitlines()
    defines = [k for k, line in enumerate(lines) if line.startswith("#define RGPOT_NLIST_API")]
    if not defines:
        raise SystemExit(f"{HEADER}: no #define RGPOT_NLIST_API")
    start = next(k for k in range(defines[0], -1, -1) if lines[k].startswith("#if"))
    end = next(k for k in range(defines[-1], len(lines)) if lines[k].startswith("#endif"))
    return "\n".join(lines[start : end + 1])


def preprocessor() -> list[str]:
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return [name, "-E", "-P", "-x", "c"]
    raise SystemExit("no C preprocessor found (set CC)")


def expand(block: str, defines: list[str]) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "macro.h"
        src.write_text(f"{block}\n{MARK} RGPOT_NLIST_API {MARK}\n", encoding="utf-8")
        out = subprocess.run(
            preprocessor() + defines + [str(src)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    for line in out.splitlines():
        if MARK in line:
            return line.split(MARK)[1].strip()
    raise SystemExit("the preprocessor dropped the marker line")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    text = (root / HEADER).read_text(encoding="utf-8")
    block = macro_block(text)
    failures = []
    for label, defines, want in CASES:
        got = expand(block, defines)
        if got.replace(" ", "") != want.replace(" ", ""):
            failures.append(f"{label}: RGPOT_NLIST_API is {got!r}, want {want!r}")
    if "#undef RGPOT_NLIST_API" not in text:
        failures.append("Neighbours.hpp does not #undef RGPOT_NLIST_API")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(f"ok {len(CASES)} expansions of RGPOT_NLIST_API")
    return 0


if __name__ == "__main__":
    sys.exit(main())
