#!/usr/bin/env python3
"""Fail if a public C function is absent from the org-mode book.

Headers scanned: the core C ABI, the neighbour query, the fast pair
list, the plugin descriptor, and the engine ABI. A typedef whose name
ends in ``_t`` is not a function. The book is every ``*.org`` file
under ``docs/orgmode``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADERS = (
    ROOT / "rgpot-core/include/rgpot.h",
    ROOT / "CppCore/rgpot/nlist/rgpot_nlist.h",
    ROOT / "CppCore/rgpot/nlist/fastnlist.h",
    ROOT / "include/rgpot/plugin.h",
    ROOT / "CppCore/rgpot/engine_c_abi.h",
)


def functions() -> list[str]:
    names: set[str] = set()
    for header in HEADERS:
        text = header.read_text(encoding="utf-8")
        for match in re.finditer(r"\brgpot_[A-Za-z0-9_]+(?=\s*\()", text):
            name = match.group(0)
            if not name.endswith("_t"):
                names.add(name)
    return sorted(names)


def main() -> int:
    book = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "docs/orgmode").rglob("*.org")
    )
    missing = [name for name in functions() if name not in book]
    if missing:
        print("undocumented C functions:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1
    print(f"{len(functions())} C functions are named in the book")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
