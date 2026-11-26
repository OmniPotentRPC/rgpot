#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""The vendored orcac.h matches its pin, and orcac's own header when given.

The pin names the orcac commit the header was copied from, and
subprojects/orcac.wrap must fetch that same commit. Set ORCAC_SOURCE_DIR to
an orcac checkout to diff the vendored copy against its include/orcac.h;
without it the wrap's directory under subprojects is used when the wrap has
been fetched, and otherwise only the pin is checked.
"""

from __future__ import annotations

import configparser
import difflib
import hashlib
import os
import re
import sys
from pathlib import Path

MARKER = "/* ---- orcac include/orcac.h ---- */\n"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def vendored_body(header: Path) -> str:
    text = header.read_text(encoding="utf-8")
    require(text.count(MARKER) == 1, f"{header} must hold the marker line once")
    return text.split(MARKER, 1)[1]


def read_pin(pin: Path) -> dict[str, str]:
    fields = {}
    for line in pin.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    orcac_dir = root / "CppCore" / "rgpot" / "ORCACPot"
    body = vendored_body(orcac_dir / "orcac.h")
    pin = read_pin(orcac_dir / "orcac.h.pin")

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    require(
        digest == pin.get("sha256"),
        f"orcac.h changed without its pin: sha256 {digest}, pin {pin.get('sha256')}",
    )
    abi = re.search(r"^#define ORCAC_ABI_VERSION (\d+)\b", body, re.MULTILINE)
    require(abi is not None, "orcac.h defines no ORCAC_ABI_VERSION")
    require(
        abi.group(1) == pin.get("abi"),
        f"ORCAC_ABI_VERSION {abi.group(1)} but the pin says {pin.get('abi')}",
    )
    require(
        pin.get("state") == "frozen",
        f"orcac.h pin state is {pin.get('state')!r}; rgpot builds on a frozen ABI",
    )

    wrap = configparser.ConfigParser()
    wrap.read(root / "subprojects" / "orcac.wrap", encoding="utf-8")
    revision = wrap.get("wrap-git", "revision", fallback=None)
    require(
        revision == pin.get("commit"),
        f"subprojects/orcac.wrap fetches {revision}, the pin names {pin.get('commit')}",
    )

    source = os.environ.get("ORCAC_SOURCE_DIR")
    tree = root / "subprojects" / wrap.get("wrap-git", "directory", fallback="orcac")
    if not source and (tree / "include" / "orcac.h").is_file():
        source = str(tree)
    if source:
        upstream = (Path(source) / "include" / "orcac.h").read_text(encoding="utf-8")
        if upstream != body:
            diff = difflib.unified_diff(
                upstream.splitlines(keepends=True),
                body.splitlines(keepends=True),
                fromfile="orcac/include/orcac.h",
                tofile="rgpot/CppCore/rgpot/ORCACPot/orcac.h",
            )
            sys.stdout.writelines(diff)
            raise AssertionError(f"vendored orcac.h differs from {source}")
        print("ok: vendored orcac.h equals", source)
    print(f"ok: orcac.h matches its pin ({pin.get('state')}, abi {pin.get('abi')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
