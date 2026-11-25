#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""rgpot dlopens the split orcac library. No ORCA code, no orcac embed."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    orca_dir = root / "CppCore" / "rgpot" / "ORCAPot"
    meson = (orca_dir / "meson.build").read_text(encoding="utf-8")
    frontend = (orca_dir / "ORCAPot.cc").read_text(encoding="utf-8")

    header = (orca_dir / "orcac.h").read_text(encoding="utf-8")
    for token in ("shared_library(", "liborcac.so", "orcac_energy_gradient"):
        require(token not in meson, f"meson.build must not embed orcac ({token})")
    require("DynLib" in frontend, "frontend must dlopen")
    require('#include "rgpot/ORCAPot/orcac.h"' in frontend, "frontend must use the vendored orcac.h")
    require(
        '{"ORCAC_LIBRARY", "RGPOT_ORCAC_ENGINE"}' in frontend,
        "frontend must search ORCAC_LIBRARY then RGPOT_ORCAC_ENGINE",
    )
    require(
        ".exclusive_explicit = true" in frontend and ".exclusive_env = true" in frontend,
        "a named library or variable must be the only path tried",
    )
    declared = re.findall(r"^ORCAC_API [^(]*?\b(orcac_\w+)\(", header, re.MULTILINE)
    require(declared, "orcac.h declares no ORCAC_API functions")
    unresolved = [name for name in declared if f'"{name}"' not in frontend]
    require(not unresolved, f"frontend must resolve every orcac.h function: {unresolved}")
    first_other = min(
        frontend.index(f'"{name}"')
        for name in declared
        if name not in ("orcac_c_abi_version", "orcac_abi_version")
    )
    for name in ("orcac_c_abi_version", "orcac_abi_version"):
        require(
            frontend.index(f'"{name}"') < first_other,
            f"{name} must be checked before the other symbols are resolved",
        )
    for token in ("ORCAC_CONFIG_MIN_SIZE", "ORCAC_RESULT_MIN_SIZE"):
        require(token in frontend, f"the load must check reported sizes against {token}")
    require("orcapot_dep = declare_dependency(" in meson, "missing orcapot_dep")
    print("ok: orcac is a split engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
