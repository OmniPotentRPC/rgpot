#!/usr/bin/env python3
"""Fail when a virtual is added to the covered classes without updating the
counters in rgpot/abi/CxxLayout.hpp."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "rgpot"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def class_body(text: str, header: str) -> str:
    match = re.search(header, text)
    if not match:
        raise SystemExit(f"class not found: {header}")
    depth, i = 0, match.end() - 1
    start = i
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1


def virtuals(body: str, guarded: bool) -> int:
    """Count `virtual` declarations; guarded selects those inside
    `#ifdef RGPOT_HAS_CACHE`."""
    count, in_cache, depth = 0, False, 0
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#ifdef RGPOT_HAS_CACHE"):
            in_cache, depth = True, 1
            continue
        if in_cache:
            if stripped.startswith("#if"):
                depth += 1
            elif stripped.startswith("#endif"):
                depth -= 1
                if depth == 0:
                    in_cache = False
                continue
        if re.search(r"\bvirtual\b", line) and in_cache == guarded:
            count += 1
    return count


def constant(text: str, name: str) -> int:
    match = re.search(rf"{name}\s*=\s*(\d+)", text)
    if not match:
        raise SystemExit(f"constant not found: {name}")
    return int(match.group(1))


def main() -> int:
    potential = strip_comments((ROOT / "Potential.hpp").read_text())
    layout = strip_comments((ROOT / "abi" / "CxxLayout.hpp").read_text())
    base = class_body(potential, r"class\s+PotentialBase\s*\{")
    derived = class_body(
        potential, r"template\s*<typename\s+Derived>\s*class\s+Potential\b[^{]*\{"
    )
    # Overrides of base virtuals inside Potential<Derived> are not new slots,
    # so only `virtual` declarations without `override` count there.
    own = sum(
        1
        for decl in re.findall(r"[^;{}]*\bvirtual\b[^;{]*[;{]", derived)
        if not re.search(r"\boverride\b", decl)
    )
    found = {
        "kPotentialBaseVirtuals": virtuals(base, guarded=False),
        "kPotentialBaseCacheVirtuals": virtuals(base, guarded=True),
        "kPotentialVirtuals": own,
    }
    failures = []
    for name, value in found.items():
        want = constant(layout, name)
        print(f"{name}: header {value}, constant {want}")
        if value != want:
            failures.append(name)
    if failures:
        print(
            "virtual count changed: update the constant and increment "
            "kCxxLayoutRevision in CppCore/rgpot/abi/CxxLayout.hpp",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
