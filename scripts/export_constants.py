#!/usr/bin/env python3
"""Check the fdlibm exp constants against the Rust bits and the golden file."""

import json
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADER = ROOT / "CppCore/rgpot/simd/exp.hpp"
RUST = ROOT / "rgpot-core/src/fp_constants.rs"
GOLDEN = ROOT / "validation/golden/exp_constants.json"

NAMES = {
    "kLn2Hi": "LN2_HI_BITS",
    "kLn2Lo": "LN2_LO_BITS",
    "kInvLn2": "INV_LN2_BITS",
    "kP1": "P1_BITS",
    "kP2": "P2_BITS",
    "kP3": "P3_BITS",
    "kP4": "P4_BITS",
    "kP5": "P5_BITS",
    "kOverflow": "OVERFLOW_BITS",
    "kUnderflow": "UNDERFLOW_BITS",
}


def header_bits() -> dict[str, str]:
    text = HEADER.read_text()
    found = {}
    for line in text.splitlines():
        line = line.strip()
        for cname, rname in NAMES.items():
            key = f"{cname} = "
            if key in line and "constexpr" in line:
                decimal = line.split(key, 1)[1].split(";")[0].strip()
                value = float(decimal)
                bits = struct.unpack("<Q", struct.pack("<d", value))[0]
                found[rname] = f"{bits:016x}"
    missing = [n for n in NAMES.values() if n not in found]
    if missing:
        raise SystemExit(f"header missing {missing}")
    return found


def rust_bits() -> dict[str, str]:
    found = {}
    for line in RUST.read_text().splitlines():
        line = line.strip()
        for rname in NAMES.values():
            if line.startswith(f"pub const {rname}"):
                hexpart = line.split("0x", 1)[1].split(";")[0].strip()
                found[rname] = f"{int(hexpart, 16):016x}"
    missing = [n for n in NAMES.values() if n not in found]
    if missing:
        raise SystemExit(f"rust missing {missing}")
    return found


def main() -> None:
    got = header_bits()
    rust = rust_bits()
    if got != rust:
        raise SystemExit(f"header and rust differ: {got} vs {rust}")
    write = "--write" in sys.argv
    check = "--check" in sys.argv or not write
    if write:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=2) + "\n")
    if check and GOLDEN.is_file():
        golden = json.loads(GOLDEN.read_text())
        if golden != got:
            raise SystemExit("golden differs from the header")
    elif check and not GOLDEN.is_file():
        raise SystemExit(f"missing {GOLDEN}")
    print("export_constants ok")


if __name__ == "__main__":
    main()
