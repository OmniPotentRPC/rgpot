#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Check that an engine library only exports the expected ABI symbols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abi_symbols import pe_exports  # noqa: E402


def exported_symbols(library: Path) -> set[str]:
    with library.open("rb") as handle:
        if handle.read(2) == b"MZ":
            return set(pe_exports(library))
    if sys.platform == "darwin":
        command = ["nm", "-gU", str(library)]
        prefix = "_"
    else:
        command = ["nm", "-D", "--defined-only", str(library)]
        prefix = ""
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    names: set[str] = set()
    ignored = {
        "_init",
        "_fini",
        "__bss_start",
        "_edata",
        "_end",
        "__dso_handle",
        "__gmon_start__",
        "__TMC_END__",
        "_ITM_deregisterTMCloneTable",
        "_ITM_registerTMCloneTable",
        "__cxa_finalize",
    }
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        symbol = parts[2]
        if prefix and symbol.startswith(prefix):
            symbol = symbol[len(prefix) :]
        if symbol in ignored:
            continue
        names.add(symbol)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("libraries", nargs="+", type=Path)
    parser.add_argument("expected", type=Path)
    args = parser.parse_args()

    expected = {
        line.strip()
        for line in args.expected.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    failed = False
    for library in args.libraries:
        exported = exported_symbols(library)
        unexpected = sorted(exported - expected)
        missing = sorted(expected - exported)
        if missing:
            failed = True
            print(f"{library}: missing symbols:", file=sys.stderr)
            for symbol in missing:
                print(f"  {symbol}", file=sys.stderr)
        if unexpected:
            failed = True
            print(f"{library}: unexpected symbols:", file=sys.stderr)
            for symbol in unexpected:
                print(f"  {symbol}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
