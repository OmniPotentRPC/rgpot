"""Frozen demanglings. The v0 strings match rustc-demangle's {:#} form."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demangle import demangle  # noqa: E402


class DemangleTest(unittest.TestCase):
    def test_rust_v0_paths(self) -> None:
        cases = {
            "_RNvC6_123foo3bar": "123foo::bar",
            "_RC4f128": "f128",
            "_RNvNtNtCs1234_7mycrate3foo3bar3baz": "mycrate::foo::bar::baz",
            "_RNvNvCs1234_7mycrate4QUUX3FOO": "mycrate::QUUX::FOO",
            "_RINvNtC3std3mem8align_ofdE": "std::mem::align_of::<f64>",
            "_RNvNvMCs1234_7mycrateINtCs1234_7mycrate3FoopE3bar4QUUX": "<mycrate::Foo<_>>::bar::QUUX",
            "_RNvNvXCs1234_7mycrateINtCs1234_7mycrate3FoopENtNtC3std5clone5Clone5clone4QUUX": (
                "<mycrate::Foo<_> as std::clone::Clone>::clone::QUUX"
            ),
            "_RNCNCNgCs6DXkGYLi8lr_2cc5spawn00B5_": "cc::spawn::{closure#0}::{closure#0}",
        }
        for mangled, plain in cases.items():
            with self.subTest(mangled=mangled):
                self.assertEqual(demangle(mangled), plain)

    def test_rust_legacy_drops_hash(self) -> None:
        self.assertEqual(
            demangle("_ZN3foo3bar17h0123456789abcdefE"),
            "foo::bar",
        )

    def test_cxx_itanium(self) -> None:
        self.assertEqual(demangle("_ZN3hwy6N_AVX34FuncEv"), "hwy::N_AVX3::Func")

    def test_plain_c_symbol(self) -> None:
        self.assertEqual(demangle("kern_avx512"), "kern_avx512")


if __name__ == "__main__":
    unittest.main()
