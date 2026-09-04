#!/usr/bin/env python3
"""Regenerate committed s-dftd3 / dftd4 golden masters. rg.terra only.

Do not invoke from meson test. Do not invent looser tolerances.
Pins are library-native: energy in Hartree, gradient in Hartree/Bohr.
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

TERRA_HOSTS = {"rgam5terra", "rg.terra"}
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "dftd"
FUNCTIONAL = "pbe"

# Water internal geometry (Angstrom), same monomer as D3PotTest / D4PotTest.
WATER = (
    (8, 0.00000000, 0.00000000, 0.11779000),
    (1, 0.00000000, 0.75545000, -0.47116000),
    (1, 0.00000000, -0.75545000, -0.47116000),
)
# (H2O)8 cube: eight monomers translated to cube vertices. Half-side 1.4 A
# gives an O-O cube edge of 2.8 A. ATM three-body is nonzero on this pin.
CUBE_HALF_A = 1.4

# s-dftd3 test/unit/test_dftd3.f90 and dftd4 test/unit/test_dftd4.f90:
#   thr  = 100*epsilon(1.0_wp)
#   thr2 = sqrt(epsilon(1.0_wp))
ENERGY_THR = 100.0 * sys.float_info.epsilon
GRAD_THR = sys.float_info.epsilon**0.5

Z_TO_SYM = {1: "H", 8: "O"}

REQUIRED = (
    "geometry.npz",
    "water_octamer.xyz",
    "d3_bj_pbe_atm_off_energy.npy",
    "d3_bj_pbe_atm_off_grad.npy",
    "d3_bj_pbe_atm_on_energy.npy",
    "d3_bj_pbe_atm_on_grad.npy",
    "d4_pbe_energy.npy",
    "d4_pbe_grad.npy",
)


def _require_terra() -> None:
    host = socket.gethostname().split(".")[0]
    if host not in TERRA_HOSTS and os.environ.get("DFTD_ALLOW_REGEN") != "1":
        sys.exit(f"regen_dftd_goldens.py runs on rg.terra only (got {host})")


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


def cube_geometry() -> tuple[list[int], list[float]]:
    numbers: list[int] = []
    pos: list[float] = []
    for sx in (-CUBE_HALF_A, CUBE_HALF_A):
        for sy in (-CUBE_HALF_A, CUBE_HALF_A):
            for sz in (-CUBE_HALF_A, CUBE_HALF_A):
                for z, x, y, zz in WATER:
                    numbers.append(z)
                    pos.extend((x + sx, y + sy, zz + sz))
    return numbers, pos


def write_xyz(path: Path, numbers: list[int], pos: list[float]) -> None:
    nat = len(numbers)
    lines = [str(nat), "water octamer cube (H2O)8, O-O edge 2.8 A"]
    for i, z in enumerate(numbers):
        x, y, zz = pos[3 * i : 3 * i + 3]
        lines.append(f"{Z_TO_SYM[z]:<2} {x:16.10f} {y:16.10f} {zz:16.10f}")
    path.write_text("\n".join(lines) + "\n")


def find_dump() -> Path:
    env = os.environ.get("DUMP_DFTD_GOLDENS")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        sys.exit(f"DUMP_DFTD_GOLDENS is not a file: {p}")
    candidates = [
        ROOT / "bbdir-dftd" / "CppCore" / "dump_dftd_goldens",
        ROOT / "bbdir-dftd" / "dump_dftd_goldens",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    sys.exit(
        "dump_dftd_goldens missing; meson compile -C bbdir-dftd dump_dftd_goldens "
        "or set DUMP_DFTD_GOLDENS"
    )


def write_manifest() -> None:
    files: dict[str, dict] = {}
    for name in REQUIRED:
        path = DATA / name
        if not path.is_file():
            sys.exit(f"missing pin after regen: {path}")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    man = {
        "host_only": "rg.terra",
        "regen": "scripts/regen_dftd_goldens.py",
        "functional": FUNCTIONAL,
        "fixture": "water_octamer_cube",
        "units": {"energy": "hartree", "gradient": "hartree/bohr"},
        "s_dftd3": "1.5.0",
        "dftd4": "4.2.0",
        "tolerances": {
            "energy": ENERGY_THR,
            "gradient": GRAD_THR,
            "source": "s-dftd3 test/unit/test_dftd3.f90 and dftd4 test/unit/test_dftd4.f90: thr=100*epsilon, thr2=sqrt(epsilon)",
        },
        "cases": [
            "d3_bj_pbe_atm_off",
            "d3_bj_pbe_atm_on",
            "d4_pbe",
        ],
        "files": files,
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print("MANIFEST.json")


def load_npy_scalar(path: Path) -> float:
    raw = path.read_bytes()
    if raw[:6] != b"\x93NUMPY":
        sys.exit(f"not npy: {path}")
    hlen = struct.unpack_from("<H", raw, 8)[0]
    payload = raw[10 + hlen :]
    return struct.unpack_from("<d", payload)[0]


def main() -> int:
    _require_terra()
    DATA.mkdir(parents=True, exist_ok=True)
    numbers, pos = cube_geometry()
    nat = len(numbers)
    write_xyz(DATA / "water_octamer.xyz", numbers, pos)
    save_npz(
        DATA / "geometry.npz",
        {
            "numbers": ([float(z) for z in numbers], [nat]),
            "positions": (pos, [nat, 3]),
        },
    )
    dump = find_dump()
    subprocess.run([str(dump)], cwd=ROOT, check=True)
    e_off = load_npy_scalar(DATA / "d3_bj_pbe_atm_off_energy.npy")
    e_on = load_npy_scalar(DATA / "d3_bj_pbe_atm_on_energy.npy")
    if abs(e_on - e_off) <= ENERGY_THR:
        sys.exit(f"ATM on == ATM off on pin ({e_on!r} vs {e_off!r})")
    write_manifest()
    print(f"d3 atm off {e_off:.16e} Eh")
    print(f"d3 atm on  {e_on:.16e} Eh")
    print(f"d4 pbe     {load_npy_scalar(DATA / 'd4_pbe_energy.npy'):.16e} Eh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
