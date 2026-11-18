#!/usr/bin/env python3
"""Require additive rgpot-core C ABI exports to advance the ABI minor."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"ABI minor check failed: {message}")


def match_int(pattern: str, text: str, source: Path) -> int:
    match = re.search(pattern, text)
    if match is None:
        fail(f"cannot read the ABI minor from {source}")
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    version = root / "rgpot-core/src/version.rs"
    header = args.header or root / "rgpot-core/include/rgpot.h"
    docs = root / "docs/orgmode/reference/c_abi.org"
    baseline = args.baseline or root / "scripts/abi_core_baseline.txt"
    ledger = args.ledger or root / "scripts/abi_core_minor_symbols.txt"

    rust_minor = match_int(
        r"RGPOT_ABI_VERSION_MINOR:\s*u16\s*=\s*(\d+)", version.read_text(), version
    )
    header_text = header.read_text()
    header_minor = match_int(
        r"#define\s+RGPOT_ABI_VERSION_MINOR\s+(\d+)", header_text, header
    )
    docs_minor = match_int(
        r"against:\s+\d+,\s+(\d+)\s+and\s+\d+\.", docs.read_text(), docs
    )
    if len({rust_minor, header_minor, docs_minor}) != 1:
        fail(
            "minor records disagree: "
            f"version.rs={rust_minor}, rgpot.h={header_minor}, c_abi.org={docs_minor}"
        )

    ledger_lines = [
        line.strip()
        for line in ledger.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    ledger_minor = match_int(r"minor=(\d+)", ledger_lines.pop(0), ledger)
    recorded_symbols = set(ledger_lines)

    without_comments = re.sub(r"/\*.*?\*/", "", header_text, flags=re.DOTALL)
    header_symbols = set(
        re.findall(r"\b(rgpot_[A-Za-z0-9_]+)\s*\(", without_comments)
    )
    baseline_symbols = {
        line.strip()
        for line in baseline.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    current_symbols = header_symbols | baseline_symbols

    removed = sorted(recorded_symbols - current_symbols)
    if removed:
        fail("recorded symbols disappeared: " + ", ".join(removed))
    added = sorted(current_symbols - recorded_symbols)
    if added and rust_minor <= ledger_minor:
        fail(
            f"new symbols require a minor above {ledger_minor}: " + ", ".join(added)
        )
    if rust_minor < ledger_minor:
        fail(f"minor {rust_minor} is below recorded minor {ledger_minor}")

    print(
        f"C ABI minor {rust_minor} covers {len(current_symbols)} header/baseline symbols"
    )


if __name__ == "__main__":
    main()
