#!/usr/bin/env python3
"""Clear PT_GNU_STACK PF_X inside an AOTI .pt2 wrapper.so.

Elja inductor emits wrapper.so with an executable stack. Terra (glibc
2.41+, hardened kernel) refuses mprotect(PROT_EXEC) on the stack, so
AOTIModelPackageLoader dies in dlopen. Terra-traced packages already
have RW GNU_STACK. This rewrite makes an Elja package load there.
"""

from __future__ import annotations

import argparse
import struct
import sys
import zipfile
from pathlib import Path

PT_GNU_STACK = 0x6474E551
PF_X = 1


def clear_elf_gnu_stack(blob: bytes) -> bytes:
    """Drop PT_GNU_STACK PF_X on an ELF64 little-endian shared object."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return blob
    data = bytearray(blob)
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    if phentsize < 8 or phnum == 0:
        return blob
    for i in range(phnum):
        off = phoff + i * phentsize
        if off + 8 > len(data):
            break
        p_type, p_flags = struct.unpack_from("<II", data, off)
        if p_type == PT_GNU_STACK and p_flags & PF_X:
            struct.pack_into("<I", data, off + 4, p_flags & ~PF_X)
    return bytes(data)


def elf_needs_gnu_stack_clear(blob: bytes) -> bool:
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4] != 2 or blob[5] != 1:
        return False
    phoff = struct.unpack_from("<Q", blob, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", blob, 54)
    if phentsize < 8 or phnum == 0:
        return False
    for i in range(phnum):
        off = phoff + i * phentsize
        if off + 8 > len(blob):
            break
        p_type, p_flags = struct.unpack_from("<II", blob, off)
        if p_type == PT_GNU_STACK:
            return bool(p_flags & PF_X)
    return False


def clear_pt2_execstack(path: Path) -> bool:
    """Rewrite wrapper.so GNU_STACK in a .pt2. Returns True if a bit cleared."""
    path = Path(path)
    changed = False
    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        payloads = {}
        for name in names:
            raw = zin.read(name)
            if name.endswith(".so"):
                patched = clear_elf_gnu_stack(raw)
                if patched != raw:
                    changed = True
                    raw = patched
            payloads[name] = raw
        infos = {info.filename: info for info in zin.infolist()}
    if not changed:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zout:
        for name in names:
            info = zipfile.ZipInfo(filename=name, date_time=infos[name].date_time)
            info.compress_type = zipfile.ZIP_STORED
            zout.writestr(info, payloads[name])
    tmp.replace(path)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pt2", nargs="+", type=Path, help="AOTI .pt2 packages")
    args = p.parse_args(argv)
    n = 0
    for path in args.pt2:
        if clear_pt2_execstack(path):
            print(f"cleared {path}")
            n += 1
        else:
            print(f"clean {path}")
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
