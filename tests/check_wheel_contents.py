"""Fail when a built rgpot wheel carries the xckernel Python generator or sympy.

The xckernel generator (catalog.py, cbackend.py) is a build-time tool. The
wheel ships only the compiled kernels, so neither the generator sources nor a
sympy dependency may reach an installed environment.

Usage: python tests/check_wheel_contents.py WHEEL [WHEEL ...]
Exit status is non-zero on any violation, and when no wheel is given.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import PurePosixPath

GENERATOR_FILES = frozenset({"catalog.py", "cbackend.py"})
SYMPY_IMPORT = re.compile(rb"^\s*(?:import|from)\s+sympy\b", re.MULTILINE)
SYMPY_REQUIREMENT = re.compile(r"^Requires-Dist:\s*sympy\b", re.IGNORECASE)


def violations(wheel: str) -> list[str]:
    """Return one message per forbidden member or dependency in `wheel`."""
    found: list[str] = []
    with zipfile.ZipFile(wheel) as zf:
        for name in zf.namelist():
            path = PurePosixPath(name)
            if path.name in GENERATOR_FILES:
                found.append(f"generator source in wheel: {name}")
            if "sympy" in path.parts:
                found.append(f"sympy package in wheel: {name}")
            if path.name == "METADATA" and path.parent.name.endswith(".dist-info"):
                for line in zf.read(name).decode("utf-8", "replace").splitlines():
                    if SYMPY_REQUIREMENT.match(line):
                        found.append(f"sympy requirement in {name}: {line.strip()}")
            elif path.suffix == ".py" and SYMPY_IMPORT.search(zf.read(name)):
                found.append(f"module imports sympy: {name}")
    return found


def main(argv: list[str]) -> int:
    wheels = argv[1:]
    if not wheels:
        print("check_wheel_contents: no wheel given", file=sys.stderr)
        return 2
    status = 0
    for wheel in wheels:
        bad = violations(wheel)
        print(f"== {wheel}: {'FAIL' if bad else 'ok'}")
        for msg in bad:
            print(f"   {msg}", file=sys.stderr)
        status |= bool(bad)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
