# Copyright 2023--present rgpot developers
"""abi_symbols.pe_exports on minimal PE32+ images built in memory."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from abi_symbols import pe_exports

PE = 0x40
OPT_SIZE = 0xF0
SECTION_RVA = 0x1000
SECTION_RAW = 0x200


def _image(names: list[str]) -> bytes:
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, PE)
    data[PE : PE + 4] = b"PE\0\0"
    struct.pack_into("<H", data, PE + 6, 1)
    struct.pack_into("<H", data, PE + 20, OPT_SIZE)
    opt = PE + 24
    struct.pack_into("<H", data, opt, 0x20B)
    struct.pack_into("<II", data, opt + 112, SECTION_RVA, 40)
    struct.pack_into(
        "<8xIIII", data, opt + OPT_SIZE, 0x200, SECTION_RVA, 0x200, SECTION_RAW
    )
    names_rva = SECTION_RVA + 0x40 if names else 0
    struct.pack_into("<24xI4xI", data, SECTION_RAW, len(names), names_rva)
    string_rva = SECTION_RVA + 0x80
    for i, name in enumerate(names):
        struct.pack_into("<I", data, SECTION_RAW + 0x40 + 4 * i, string_rva)
        start = SECTION_RAW + string_rva - SECTION_RVA
        data[start : start + len(name) + 1] = name.encode("ascii") + b"\0"
        string_rva += len(name) + 1
    return bytes(data)


def test_named_exports(tmp_path):
    dll = tmp_path / "named.dll"
    dll.write_bytes(_image(["rgpot_a", "rgpot_b"]))
    assert pe_exports(dll) == ["rgpot_a", "rgpot_b"]


def test_export_table_with_no_names(tmp_path):
    dll = tmp_path / "empty.dll"
    dll.write_bytes(_image([]))
    assert pe_exports(dll) == []
