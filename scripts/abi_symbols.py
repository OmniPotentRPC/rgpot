#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Print the stable C symbols of a library, one per line.

Usage: abi_symbols.py [-p PATTERN] [-l LIB]... LIB [BASELINE...]

The default names are every rgpot_ export. abi_baseline.txt is every
function rgpot/rgpot_api.h marks, and those are always exported.
-p replaces the name pattern, a regular expression matched from the
start of each name.
-l adds a library whose symbols count with those of LIB. A static
librgpot leaves the neighbour list in its own archive, and a consumer
links both, so the build passes that archive with -l and checks the
union against one baseline.

Each baseline lists names the library must export. The build passes the
fast-list baseline only when it links the fast list. Engine and RPC
names appear only in the builds that include them. A missing baseline
name fails. An extra name does not: it is an addition.

A DLL exports what its export table lists. The table is read from the
PE file itself, so the check needs neither dumpbin nor objdump. Mach-O
nm has no -D and prefixes C names with an underscore. A static archive
(lib*.a, or *.lib from an MSVC-style toolchain) has no dynamic symbol
table, so nm -D finds nothing in it; its defined external symbols are
read instead. A shared object keeps nm -D, so hidden symbols stay out.
A failing nm fails the check.

Shared libraries may not add C++ names in an internal namespace. Those
namespaces are rgpot::detail and every rgpot namespace named detail or
detail_fill (nlist, simd, abi, ipi, aoti_unpack, and the same kind of
helper elsewhere under rgpot). Public component namespaces are the C++
API and are not part of this check. pocketfft::detail belongs to that
third party header, so this check leaves it out. The qualified names
already exported are recorded in abi_detail_baseline.txt, without
parameter lists: a libstdc++ and a libc++ demangle of the same function
do not spell std::string the same way, and the macOS build would
otherwise report those signatures as additions. A new overload of a
name already in the baseline is not distinguished. A weak out of line
copy of an inline (nm type W or V) is omitted: a debug build emits it
and a release build does not, so it is not a stable addition. A DLL
export table has no symbol types, so the detail check does not read a
DLL.

Names are compared and printed in byte order, so the printed list is
the same on every machine.
"""
from __future__ import annotations

import re
import struct
import subprocess
import sys
from pathlib import Path

DETAIL_BASELINE = Path(__file__).resolve().parent / "abi_detail_baseline.txt"


def fail(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.exit(1)


def pe_exports(path: Path) -> list[str]:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        fail(f"{path} is not a PE file")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        fail(f"{path} has no PE signature")
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
        fail(f"{path}: RVA {rva:#x} is in no section")
        return 0

    ed = offset(export_rva)
    count, names_rva = struct.unpack_from("<24xI4xI", data, ed)
    names = offset(names_rva)
    out = []
    for i in range(count):
        start = offset(struct.unpack_from("<I", data, names + 4 * i)[0])
        out.append(data[start : data.index(b"\0", start)].decode("ascii"))
    return out


def is_archive(path: Path) -> bool:
    return path.suffix in (".a", ".lib")


def nm_symbols(path: Path) -> list[tuple[str, str]]:
    """(type, name) of every defined global symbol nm reports."""
    darwin = sys.platform == "darwin"
    if darwin:
        cmd = ["nm", "-gU", str(path)]
    elif is_archive(path):
        cmd = ["nm", "-g", "--defined-only", str(path)]
    else:
        cmd = ["nm", "-D", "--defined-only", str(path)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as err:
        fail(f"cannot run nm: {err}")
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        fail(f"nm failed on {path}")
    out = []
    for line in run.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        name = fields[2]
        if darwin and name.startswith("_"):
            name = name[1:]
        out.append((fields[1], name))
    return out


def qualified(name: str) -> str:
    name = name.replace("[abi:cxx11]", "")
    paren = name.find("(")
    return name if paren < 0 else name[:paren]


def detail_names(symbols: list[tuple[str, str]]) -> set[str]:
    # W and V are weak defined symbols. Inline helpers show up that way
    # when the compiler emits an out-of-line copy, which depends on the
    # optimisation level.
    mangled = [name for kind, name in symbols if kind not in ("W", "V")]
    if not mangled:
        return set()
    try:
        run = subprocess.run(
            ["c++filt"],
            input="\n".join(mangled) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as err:
        fail(f"cannot run c++filt: {err}")
    if run.returncode != 0:
        sys.stderr.write(run.stderr)
        fail("c++filt failed")
    out = set()
    for line in run.stdout.splitlines():
        name = qualified(line)
        if not name.startswith("rgpot::"):
            continue
        if "::detail::" in name or "::detail_fill::" in name:
            out.add(name)
    return out


def read_lines(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main() -> None:
    args = sys.argv[1:]
    pattern = "^rgpot_"
    extra: list[Path] = []
    while args and args[0] in ("-p", "-l"):
        if len(args) < 2:
            fail(f"{args[0]} needs a value\n{__doc__}")
        if args[0] == "-p":
            pattern = args[1]
        else:
            extra.append(Path(args[1]))
        args = args[2:]
    if not args:
        fail(__doc__)
    lib = Path(args[0])
    baselines = [Path(arg) for arg in args[1:]]

    names: set[str] = set()
    shared: list[tuple[str, str]] = []
    for path in [*extra, lib]:
        if path.suffix.lower() == ".dll":
            names.update(pe_exports(path))
            continue
        symbols = nm_symbols(path)
        names.update(name for _, name in symbols)
        if not is_archive(path):
            shared.extend(symbols)

    if shared:
        added = sorted(detail_names(shared) - read_lines(DETAIL_BASELINE))
        if added:
            fail("C++ detail symbols added:\n" + "\n".join(added))

    match = re.compile(pattern)
    found = sorted(name for name in names if match.search(name))
    if not baselines:
        if found:
            print("\n".join(found))
        return
    want: set[str] = set()
    for baseline in baselines:
        want.update(read_lines(baseline))
    missing = sorted(want - set(found))
    if missing:
        fail("C ABI lost symbols:\n" + "\n".join(missing))
    print(f"C ABI keeps {len(want)} baseline symbols")


if __name__ == "__main__":
    main()
