# MIT License
# Copyright 2023--present rgpot developers
"""Keep a single named export in a PE image.

A module-definition file does not remove symbols that were marked
__declspec(dllexport) in a static library. This rewrites the export
directory so the image exports one name, which is the F9 allowlist.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

_SCN_MEM_READ = 0x40000000
_SCN_CNT_INITIALIZED_DATA = 0x00000040


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _rva_to_offset(rva: int, sections: list[tuple[int, int, int, int]]) -> int:
    for virtual_address, virtual_size, raw_pointer, raw_size in sections:
        span = max(virtual_size, raw_size)
        if virtual_address <= rva < virtual_address + span:
            return raw_pointer + (rva - virtual_address)
    raise SystemExit(f"RVA {rva:#x} is not inside a section")


def _cstring(data: bytearray, offset: int) -> bytes:
    end = data.index(b"\0", offset)
    return bytes(data[offset:end])


def limit_pe_exports(path: Path, keep: str) -> None:
    """Rewrite ``path`` so its export directory contains only ``keep``."""
    data = bytearray(path.read_bytes())
    if data[:2] != b"MZ":
        raise SystemExit(f"{path} is not a PE file")
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        raise SystemExit(f"{path} is not a PE file")
    coff = e_lfanew + 4
    _machine, nsections, _timestamp, _symptr, _nsyms, optsize, _chars = struct.unpack_from(
        "<HHIIIHH", data, coff
    )
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic == 0x20B:
        dd_off = opt + 112
    elif magic == 0x10B:
        dd_off = opt + 96
    else:
        raise SystemExit(f"unsupported PE magic {magic:#x}")
    section_align = struct.unpack_from("<I", data, opt + 32)[0]
    file_align = struct.unpack_from("<I", data, opt + 36)[0]
    size_of_image_off = opt + 56
    size_of_headers = struct.unpack_from("<I", data, opt + 60)[0]
    checksum_off = opt + 64
    export_rva, _export_size = struct.unpack_from("<II", data, dd_off)
    if export_rva == 0:
        raise SystemExit(f"{path} has an empty export directory")

    sec_off = opt + optsize
    sections: list[tuple[int, int, int, int, int]] = []
    mapped: list[tuple[int, int, int, int]] = []
    for index in range(nsections):
        off = sec_off + index * 40
        _name, virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
            "<8sIIII", data, off
        )
        sections.append((off, virtual_address, virtual_size, raw_pointer, raw_size))
        mapped.append((virtual_address, virtual_size, raw_pointer, raw_size))

    exp_off = _rva_to_offset(export_rva, mapped)
    (
        _chars_e,
        _tds,
        _maj,
        _minor,
        name_rva,
        _base,
        nfuncs,
        nnames,
        addr_funcs,
        addr_names,
        addr_ords,
    ) = struct.unpack_from("<IIHHIIIIIII", data, exp_off)
    if nnames == 0 or nfuncs == 0:
        raise SystemExit(f"{path} has an empty export directory")
    names_off = _rva_to_offset(addr_names, mapped)
    ords_off = _rva_to_offset(addr_ords, mapped)
    funcs_off = _rva_to_offset(addr_funcs, mapped)
    wanted = keep.encode("ascii")
    func_rva = None
    for index in range(nnames):
        name_ptr = struct.unpack_from("<I", data, names_off + index * 4)[0]
        symbol = _cstring(data, _rva_to_offset(name_ptr, mapped))
        if symbol == wanted:
            ordinal_index = struct.unpack_from("<H", data, ords_off + index * 2)[0]
            if ordinal_index >= nfuncs:
                raise SystemExit(f"{keep} ordinal is outside the function table")
            func_rva = struct.unpack_from("<I", data, funcs_off + ordinal_index * 4)[0]
            break
    if func_rva is None:
        raise SystemExit(f"{keep} is not exported from {path}")

    dll_name = _cstring(data, _rva_to_offset(name_rva, mapped)) + b"\0"
    func_name = wanted + b"\0"
    # Directory, one function RVA, one name RVA, one ordinal, then the strings.
    strings_at = 40 + 4 + 4 + 2
    strings_at += (4 - (strings_at % 4)) % 4
    dll_at = strings_at
    func_at = dll_at + len(dll_name)
    blob_size = func_at + len(func_name)

    last_header, last_va, last_vsz, last_raw, last_rawsz = sections[-1]
    header_end = sec_off + nsections * 40
    can_add_section = size_of_headers >= header_end + 40
    if can_add_section:
        blob_va = _align(last_va + max(last_vsz, 1), section_align)
        blob_raw = _align(max(len(data), last_raw + last_rawsz), file_align)
        raw_size = _align(blob_size, file_align)
        blob = _export_blob(blob_va, dll_at, func_at, dll_name, func_name, func_rva, blob_size)
        if len(data) < blob_raw + raw_size:
            data.extend(b"\0" * (blob_raw + raw_size - len(data)))
        data[blob_raw : blob_raw + len(blob)] = blob
        struct.pack_into("<H", data, coff + 2, nsections + 1)
        struct.pack_into(
            "<8sIIIIIIHHI",
            data,
            header_end,
            b".edata\0\0",
            blob_size,
            blob_va,
            raw_size,
            blob_raw,
            0,
            0,
            0,
            0,
            _SCN_CNT_INITIALIZED_DATA | _SCN_MEM_READ,
        )
        size_of_image = _align(blob_va + blob_size, section_align)
    else:
        blob_delta = max(last_vsz, last_rawsz)
        blob_va = last_va + blob_delta
        blob_raw = last_raw + blob_delta
        blob = _export_blob(blob_va, dll_at, func_at, dll_name, func_name, func_rva, blob_size)
        if len(data) < blob_raw:
            data.extend(b"\0" * (blob_raw - len(data)))
        data[blob_raw : blob_raw + len(blob)] = blob
        end = blob_raw + len(blob)
        raw_size = _align(end - last_raw, file_align)
        if len(data) < last_raw + raw_size:
            data.extend(b"\0" * (last_raw + raw_size - len(data)))
        struct.pack_into("<I", data, last_header + 8, blob_delta + len(blob))
        struct.pack_into("<I", data, last_header + 16, raw_size)
        size_of_image = _align(blob_va + len(blob), section_align)

    struct.pack_into("<I", data, size_of_image_off, size_of_image)
    struct.pack_into("<II", data, dd_off, blob_va, blob_size)
    struct.pack_into("<I", data, checksum_off, 0)
    path.write_bytes(data)


def _export_blob(
    blob_va: int,
    dll_at: int,
    func_at: int,
    dll_name: bytes,
    func_name: bytes,
    func_rva: int,
    blob_size: int,
) -> bytearray:
    blob = bytearray(blob_size)

    def at(relative: int) -> int:
        return blob_va + relative

    struct.pack_into(
        "<IIHHIIIIIII",
        blob,
        0,
        0,
        0,
        0,
        0,
        at(dll_at),
        1,
        1,
        1,
        at(40),
        at(44),
        at(48),
    )
    struct.pack_into("<I", blob, 40, func_rva)
    struct.pack_into("<I", blob, 44, at(func_at))
    struct.pack_into("<H", blob, 48, 0)
    blob[dll_at : dll_at + len(dll_name)] = dll_name
    blob[func_at : func_at + len(func_name)] = func_name
    return blob


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <pe> <export>")
    limit_pe_exports(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
