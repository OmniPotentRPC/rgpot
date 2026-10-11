"""A hand-built Mach-O slice, so the parser is tested without a macOS SDK."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa_audit import AuditStats, audit_bytes, load_allowlist  # noqa: E402

# cntw x5 from the aarch64 fixture, little endian.
_CNTW_X5 = bytes.fromhex("e5e3a004")
_RET = bytes.fromhex("c0035fd6")


def _macho(code: bytes, name: str | None) -> bytes:
    # layout: header, segment+section, symtab, code, symbol, strings
    code_off = 32 + 72 + 80 + 24
    code_off = (code_off + 7) & ~7
    sym_off = code_off + len(code)
    str_off = sym_off + (16 if name else 0)
    # Mach-O string tables start with a NUL.
    strings = b"\x00"
    if name:
        strings += name.encode() + b"\x00"
    blob = bytearray()
    # mach_header_64, CPU_TYPE_ARM64, MH_OBJECT
    blob += struct.pack("<IIIIIIII", 0xFEEDFACF, 0x0100000C, 0, 1, 2, 72 + 80 + 24, 0, 0)
    # LC_SEGMENT_64
    sect_size = len(code)
    blob += struct.pack("<II", 0x19, 72 + 80)
    blob += struct.pack("<16s", b"__TEXT")
    blob += struct.pack("<QQQQ", 0, sect_size, code_off, sect_size)
    blob += struct.pack("<IIII", 7, 5, 1, 0)
    blob += struct.pack("<16s16s", b"__text", b"__TEXT")
    blob += struct.pack("<QQ", 0, sect_size)
    blob += struct.pack("<IIIIIIII", code_off, 2, 0, 0, 0x80000000, 0, 0, 0)
    nsyms = 1 if name else 0
    blob += struct.pack("<IIIIII", 0x02, 24, sym_off, nsyms, str_off, len(strings))
    while len(blob) < code_off:
        blob += b"\x00"
    blob += code
    if name:
        # n_strx points at the name after the leading NUL. N_SECT|N_EXT, sect 1.
        blob += struct.pack("<IBBHQ", 1, 0x0F, 1, 0, 0)
    blob += strings
    return bytes(blob)


class MachOTest(unittest.TestCase):
    def test_allowed_sve_symbol(self) -> None:
        allow = load_allowlist(Path(__file__).with_name("allow-sve.toml"))
        stats = AuditStats()
        audit_bytes(_macho(_CNTW_X5 + _RET, "_kern_sve"), "kern.o", stats, allow, "armv8-a")
        self.assertEqual(stats.violations, [])
        self.assertGreater(stats.allowed_feature_counts.get("sve", 0), 0)
        self.assertEqual(stats.files_audited, 1)

    def test_stripped_sve_is_no_symbols(self) -> None:
        allow = load_allowlist(Path(__file__).with_name("allow-sve.toml"))
        stats = AuditStats()
        audit_bytes(_macho(_CNTW_X5 + _RET, None), "stripped.o", stats, allow, "armv8-a")
        self.assertTrue(stats.violations)
        self.assertTrue(all(item.symbol == "no symbols" for item in stats.violations))
        self.assertTrue(any(item.feature == "sve" for item in stats.violations))


if __name__ == "__main__":
    unittest.main()
