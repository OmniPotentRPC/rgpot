#!/usr/bin/env python3
"""Print the stable C symbols of librgpot, one per line.

With baselines, fail on any baseline name the library does not export.
abi_baseline.txt is every function rgpot/rgpot_api.h marks, and those are
always exported. Fast-list names exist only when the fast list is built,
so meson passes abi_baseline_fast.txt only then. An extra name does not
fail: it is an addition.

A DLL exports what its export table lists; the table is read from the PE
file itself, so the Windows test needs neither dumpbin nor objdump.
Apple nm has no -D and prefixes C names with _. A static librgpot (the
macOS CI default) has no dynamic table, only member globals.

Usage: abi_symbols.py librgpot [baseline ...]
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path


def pe_exports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        sys.exit(f"{path} is not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        sys.exit(f"{path} has no PE signature")
    sections, opt_size = struct.unpack_from("<H12xH", data, pe + 6)
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dirs = opt + (112 if magic == 0x20B else 96)
    export_rva, export_size = struct.unpack_from("<II", data, dirs)
    if export_size == 0:
        return []
    table = opt + opt_size
    spans = [
        struct.unpack_from("<8xIIII", data, table + 40 * i) for i in range(sections)
    ]

    def offset(rva: int) -> int:
        for vsize, vaddr, raw_size, raw in spans:
            if vaddr <= rva < vaddr + max(vsize, raw_size):
                return raw + rva - vaddr
        sys.exit(f"{path}: RVA {rva:#x} is in no section")

    ed = offset(export_rva)
    count, names_rva = struct.unpack_from("<24xI4xI", data, ed)
    names = offset(names_rva)
    out = []
    for i in range(count):
        start = offset(struct.unpack_from("<I", data, names + 4 * i)[0])
        out.append(data[start : data.index(b"\0", start)].decode("ascii"))
    return out


def nm_symbols(path: Path) -> list[str]:
    if sys.platform == "darwin":
        cmd = ["nm", "-gU", str(path)]
    elif path.suffix == ".a":
        cmd = ["nm", "-g", "--defined-only", str(path)]
    else:
        cmd = ["nm", "-D", "--defined-only", str(path)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as err:
        sys.exit(f"cannot run nm: {err}")
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        sys.exit(f"nm failed on {path}")
    names = []
    for line in run.stdout.splitlines():
        fields = line.split()
        if fields:
            name = fields[-1]
            if sys.platform == "darwin" and name.startswith("_"):
                name = name[1:]
            names.append(name)
    return names


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    lib = Path(sys.argv[1])
    if lib.suffix.lower() == ".dll":
        symbols = pe_exports(lib)
    else:
        symbols = nm_symbols(lib)
    found = {name for name in symbols if name.startswith("rgpot_")}
    if len(sys.argv) == 2:
        print("\n".join(sorted(found)))
        return
    want = set()
    for baseline in sys.argv[2:]:
        want.update(
            line.strip() for line in Path(baseline).read_text().splitlines() if line.strip()
        )
    missing = sorted(want - found)
    if missing:
        print("C ABI lost symbols:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        sys.exit(1)
    print(f"C ABI keeps {len(want)} baseline symbols")


if __name__ == "__main__":
    main()
