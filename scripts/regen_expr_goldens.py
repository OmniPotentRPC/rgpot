#!/usr/bin/env python3
"""Regenerate committed ExprPot algebra golden masters on a regen host.

Runs only on a host listed in RGPOT_REGEN_HOSTS (comma-separated),
unless EXPR_ALLOW_REGEN=1.

Do not invoke from meson test. Do not invent looser tolerances.
Pins are independent sums: energy in eV, forces in eV/A.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "expr"

# Two-atom Ar fixture shared with ExprPotTest identity (Angstrom, eV).
POSITIONS = [0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
NUMBERS = [18, 18]
BOX = [40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0]

ENERGY_TOL = 1e-14
FORCE_TOL = 1e-12

ALWAYS_ON = (
    "geometry.npz",
    "identity_energy.npy",
    "identity_forces.npy",
    "half_lj_plus_morse_energy.npy",
    "half_lj_plus_morse_forces.npy",
    "two_lj_minus_lj_energy.npy",
    "two_lj_minus_lj_forces.npy",
    "half_paren_lj_plus_morse_energy.npy",
    "half_paren_lj_plus_morse_forces.npy",
)
OPTIONAL_D3 = (
    "half_lj_plus_d3_energy.npy",
    "half_lj_plus_d3_forces.npy",
)


def _require_regen_host() -> None:
    raw = os.environ.get("RGPOT_REGEN_HOSTS", "")
    allowed = {h.strip() for h in raw.split(",")} - {""}
    host = socket.gethostname()
    override = os.environ.get("EXPR_ALLOW_REGEN") == "1"
    if override or allowed & {host, host.split(".")[0]}:
        return
    sys.exit(
        f"regen_expr_goldens.py runs only on a host listed in RGPOT_REGEN_HOSTS "
        f"(got {host}; EXPR_ALLOW_REGEN=1 overrides)"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_npy_bytes(data: list[float], shape: list[int]) -> bytes:
    n = 1
    for d in shape:
        n *= d
    if len(data) != n:
        raise ValueError(f"save_npy size {len(data)} != {shape}")
    inner = ", ".join(str(s) for s in shape)
    if len(shape) == 1:
        inner += ","
    header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + inner + "), }"
    while (10 + len(header)) % 64 != 63:
        header += " "
    header += "\n"
    buf = bytearray()
    buf.extend(b"\x93NUMPY")
    buf.extend(bytes((1, 0)))
    buf.extend(struct.pack("<H", len(header)))
    buf.extend(header.encode("ascii"))
    buf.extend(struct.pack("<" + "d" * n, *data))
    return bytes(buf)


def save_npz(path: Path, arrays: dict[str, tuple[list[float], list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        for key, (data, shape) in arrays.items():
            zf.writestr(f"{key}.npy", save_npy_bytes(data, shape))


def write_geometry() -> None:
    save_npz(
        DATA / "geometry.npz",
        {
            "positions": (POSITIONS, [2, 3]),
            "numbers": ([float(z) for z in NUMBERS], [2]),
            "box": (BOX, [3, 3]),
        },
    )


def find_dump() -> Path:
    env = os.environ.get("DUMP_EXPR_GOLDENS")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        sys.exit(f"DUMP_EXPR_GOLDENS is not a file: {p}")
    candidates = [
        ROOT / "bbdir-expr" / "CppCore" / "dump_expr_goldens",
        ROOT / "bbdir-expr" / "dump_expr_goldens",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    sys.exit(
        "dump_expr_goldens missing; meson compile -C bbdir-expr dump_expr_goldens "
        "or set DUMP_EXPR_GOLDENS"
    )


def write_manifest() -> None:
    files: dict[str, dict] = {}
    names = list(ALWAYS_ON)
    for name in OPTIONAL_D3:
        if (DATA / name).is_file():
            names.append(name)
    for name in names:
        path = DATA / name
        if not path.is_file():
            sys.exit(f"missing pin after regen: {path}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    cases = [
        "identity",
        "half_lj_plus_morse",
        "two_lj_minus_lj",
        "half_paren_lj_plus_morse",
    ]
    if (DATA / "half_lj_plus_d3_energy.npy").is_file():
        cases.append("half_lj_plus_d3")
    man = {
        "host_only": "workstation-a",
        "regen": "scripts/regen_expr_goldens.py",
        "fixture": "two_atom_ar",
        "units": {"energy": "eV", "forces": "eV/A"},
        "tolerances": {
            "energy": ENERGY_TOL,
            "forces": FORCE_TOL,
            "source": "rgpot-2wdt: energy 1e-14 eV, forces 1e-12 eV/A",
        },
        "cases": cases,
        "files": files,
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print("MANIFEST.json")


def main() -> int:
    _require_regen_host()
    DATA.mkdir(parents=True, exist_ok=True)
    write_geometry()
    if os.environ.get("EXPR_GEOMETRY_ONLY") == "1":
        print("geometry.npz")
        return 0
    dump = find_dump()
    subprocess.run([str(dump)], cwd=ROOT, check=True)
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
