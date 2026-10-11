#!/usr/bin/env python3
"""Fail unless every C and C++ compile command turns contraction off."""

import json
import pathlib
import sys

FLAG_OK = ("-ffp-contract=off", "/fp:precise")
SOURCES = (".c", ".cc", ".cpp", ".cxx", ".C")


def command_text(entry: dict) -> str:
    if "command" in entry:
        return entry["command"]
    if "arguments" in entry:
        return " ".join(entry["arguments"])
    return ""


def source_of(entry: dict, text: str) -> str:
    if entry.get("file"):
        return entry["file"]
    parts = text.split()
    if "-c" in parts:
        i = parts.index("-c")
        if i + 1 < len(parts):
            return parts[i + 1]
    return ""


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    entries = json.loads(path.read_text())
    bad = []
    seen = 0
    for entry in entries:
        text = command_text(entry)
        src = source_of(entry, text)
        if not src.endswith(SOURCES):
            continue
        parts = text.split()
        if "-c" not in parts and "/c" not in parts:
            continue
        seen += 1
        if not any(flag in text for flag in FLAG_OK):
            bad.append(src)
    if seen == 0:
        raise SystemExit(f"no C/C++ compiles in {path}")
    if bad:
        print(f"{len(bad)} compiles lack -ffp-contract=off or /fp:precise", file=sys.stderr)
        for src in bad[:40]:
            print(f"  {src}", file=sys.stderr)
        raise SystemExit(1)
    print(f"fp-contract ok ({seen} compiles)")


if __name__ == "__main__":
    main()
