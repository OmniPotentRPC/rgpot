#!/usr/bin/env python3
"""Regenerate D3/D4 golden masters from s-dftd3 / dftd4 C APIs. rg.terra only.

Do not invoke from meson test. Do not invent looser tolerances.
Pins are Hartree / Hartree/Bohr from dump_dftd_goldens (dftd3.h / dftd4.h),
not from D3Pot/D4Pot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

TERRA_HOSTS = {"rgam5terra", "rg.terra"}
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "dftd"

# s-dftd3 test/unit/test_dftd3.f90: thr = 100*epsilon(1.0_wp) vs library refs.
# dftd4 unit tests use the same energy bar. Gradient vs finite difference is
# sqrt(eps); pins are the analytic C-API gradient, so 100*eps applies.
THR_HARTREE = 100.0 * sys.float_info.epsilon

REQUIRED = [
    "MANIFEST.json",
    "water.xyz",
    "water_pos.npy",
    "water_z.npy",
    "d3_bj_pbe_atm_off_energy.npy",
    "d3_bj_pbe_atm_off_grad.npy",
    "d3_bj_pbe_atm_on_energy.npy",
    "d3_bj_pbe_atm_on_grad.npy",
    "d4_pbe_energy.npy",
    "d4_pbe_grad.npy",
]


def _require_terra() -> None:
    host = socket.gethostname().split(".")[0]
    if host not in TERRA_HOSTS and os.environ.get("DFTD_ALLOW_REGEN") != "1":
        sys.exit(f"regen_dftd_goldens.py runs on rg.terra only (got {host})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _find_dump(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"dump binary missing: {p}")
        return p
    env = os.environ.get("DUMP_DFTD_GOLDENS")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    candidates = [
        ROOT / "bbdir-dftd" / "CppCore" / "dump_dftd_goldens",
        ROOT / "bbdir" / "CppCore" / "dump_dftd_goldens",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    sys.exit(
        "dump_dftd_goldens missing; pass --dump or set DUMP_DFTD_GOLDENS. "
        "Build with -Dwith_dftd3=true -Dwith_dftd4=true."
    )


def main() -> int:
    _require_terra()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", help="path to dump_dftd_goldens")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    dump = _find_dump(args.dump)
    proc = subprocess.run([str(dump), str(DATA)], check=False)
    if proc.returncode != 0:
        sys.exit(f"dump_dftd_goldens exited {proc.returncode}")

    files: dict[str, dict[str, int | str]] = {}
    for name in REQUIRED:
        if name == "MANIFEST.json":
            continue
        path = DATA / name
        if not path.is_file() or path.stat().st_size == 0:
            sys.exit(f"missing pin after dump: {path}")
        files[name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "regen": "scripts/regen_dftd_goldens.py",
        "dump": "CppCore/tests/dump_dftd_goldens.cc",
        "host_only": "rg.terra",
        "fixture": "baker_water",
        "functional": "pbe",
        "units": {"energy": "hartree", "gradient": "hartree/bohr"},
        "tolerances": {
            "energy_hartree": THR_HARTREE,
            "gradient_hartree_bohr": THR_HARTREE,
            "note": "100*epsilon; s-dftd3/dftd4 library energy bar",
        },
        "cases": [
            "d3_bj_pbe_atm_off",
            "d3_bj_pbe_atm_on",
            "d4_pbe",
        ],
        "files": files,
    }
    man_path = DATA / "MANIFEST.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {man_path}")
    for name in REQUIRED:
        path = DATA / name
        if not path.is_file():
            sys.exit(f"fail closed: {path} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
