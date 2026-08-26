"""PT_GNU_STACK PF_X clearer for Elja AOTI wrapper.so."""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from aoti_execstack import (  # noqa: E402
    clear_elf_gnu_stack,
    clear_pt2_execstack,
    elf_needs_gnu_stack_clear,
)


def _elf64_with_gnu_stack(flags: int) -> bytes:
    data = bytearray(120)
    data[0:4] = b"\x7fELF"
    data[4] = 2
    data[5] = 1
    data[6] = 1
    struct.pack_into("<HHI", data, 16, 3, 62, 1)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<H", data, 52, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<II", data, 64, 0x6474E551, flags)
    return bytes(data)


def test_clear_elf_drops_pf_x():
    raw = _elf64_with_gnu_stack(7)
    assert elf_needs_gnu_stack_clear(raw)
    out = clear_elf_gnu_stack(raw)
    assert not elf_needs_gnu_stack_clear(out)
    assert struct.unpack_from("<I", out, 68)[0] == 6


def test_clear_elf_idempotent_on_rw():
    raw = _elf64_with_gnu_stack(6)
    assert clear_elf_gnu_stack(raw) == raw
    assert not elf_needs_gnu_stack_clear(raw)


def test_clear_pt2_rewrites_stored_wrapper(tmp_path: Path):
    pt2 = tmp_path / "elja.pt2"
    so = _elf64_with_gnu_stack(7)
    with zipfile.ZipFile(pt2, "w") as z:
        info = zipfile.ZipInfo("model/foo.wrapper.so")
        info.compress_type = zipfile.ZIP_STORED
        z.writestr(info, so)
        z.writestr("model/data.pkl", b"meta")
    assert clear_pt2_execstack(pt2) is True
    assert clear_pt2_execstack(pt2) is False
    with zipfile.ZipFile(pt2) as z:
        blob = z.read("model/foo.wrapper.so")
    assert not elf_needs_gnu_stack_clear(blob)
    assert struct.unpack_from("<I", blob, 68)[0] == 6
