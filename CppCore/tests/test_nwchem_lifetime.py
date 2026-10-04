#!/usr/bin/env python3
"""Check engine state and finalization across public calculator lifetimes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    executable, engine = (Path(value).resolve() for value in sys.argv[1:3])
    mode = sys.argv[3]
    if mode not in {"ordinary", "probe"}:
        raise ValueError(f"unknown lifetime mode: {mode}")
    with tempfile.TemporaryDirectory(prefix="rgpot-nwchem-lifetime-") as work:
        events = Path(work) / "events.txt"
        env = os.environ.copy()
        for key in ("NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE"):
            env.pop(key, None)
        env["RGPOT_NWCHEM_ENGINE"] = str(engine)
        env["RGPOT_NWCHEM_LIFETIME_LOG"] = str(events)
        result = subprocess.run(
            [str(executable), mode], env=env, capture_output=True, text=True
        )
        observed = events.read_text().splitlines() if events.exists() else []
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        print("engine events:", observed)
        if result.returncode != 0:
            raise AssertionError(f"public lifetime driver returned {result.returncode}")
        expected = [
            "load", "call 1", "call 2", "call 3", "call 4",
            "objects destroyed", "finalize", "unload",
        ]
        if observed != expected:
            raise AssertionError(f"engine lifetime events {observed} != {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
