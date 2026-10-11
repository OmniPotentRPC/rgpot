"""Mnemonic classification for LSE atomics and baseline fcvt."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isa_audit import classify_aarch64  # noqa: E402

_OPS = ("add", "clr", "eor", "set", "smax", "smin", "umax", "umin")
_LOAD_ORDER = ("", "a", "l", "al")
_STORE_ORDER = ("", "l")
_SIZES = ("", "b", "h")


class ClassifyTest(unittest.TestCase):
    def test_lse_load_and_store_variants(self) -> None:
        for op in _OPS:
            for order in _LOAD_ORDER:
                for size in _SIZES:
                    mnemonic = f"ld{op}{order}{size}"
                    with self.subTest(mnemonic=mnemonic):
                        self.assertEqual(classify_aarch64(mnemonic, "w0, w1, [x2]"), ["lse"])
            for order in _STORE_ORDER:
                for size in _SIZES:
                    mnemonic = f"st{op}{order}{size}"
                    with self.subTest(mnemonic=mnemonic):
                        self.assertEqual(classify_aarch64(mnemonic, "w0, [x1]"), ["lse"])

    def test_lse_cas_and_swp_variants(self) -> None:
        for mnemonic in (
            "cas",
            "casa",
            "casl",
            "casal",
            "casb",
            "casab",
            "caslb",
            "casalb",
            "cash",
            "casah",
            "caslh",
            "casalh",
            "casp",
            "caspa",
            "caspl",
            "caspal",
            "swp",
            "swpa",
            "swpl",
            "swpal",
            "swpb",
            "swph",
            "swpalb",
            "swpalh",
        ):
            with self.subTest(mnemonic=mnemonic):
                self.assertEqual(classify_aarch64(mnemonic, "w0, w1, [x2]"), ["lse"])

    def test_scalar_fcvt_between_h_and_s_or_d_is_baseline(self) -> None:
        for ops in ("h0, s1", "s0, h1", "h0, d1", "d0, h1"):
            with self.subTest(ops=ops):
                self.assertEqual(classify_aarch64("fcvt", ops), [])

    def test_half_precision_arithmetic_stays_fp16(self) -> None:
        self.assertEqual(classify_aarch64("fadd", "h0, h1, h2"), ["fp16"])
        self.assertEqual(classify_aarch64("fcvtas", "h0, s1"), ["fp16"])


if __name__ == "__main__":
    unittest.main()
