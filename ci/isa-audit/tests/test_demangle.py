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
            "RNvC6_123foo3bar": "123foo::bar",
            "__RNvC6_123foo3bar": "123foo::bar",
            "_RC4f128": "f128",
            "_RNvC3std3foo": "std::foo",
            "_RINvC3std3fooE": "std::foo::<>",
            "_RNvNtC3std3mem8align_of": "std::mem::align_of",
            "_RINvNtC3std3mem8align_ofjE": "std::mem::align_of::<usize>",
            "_RINvNtC3std3mem8align_ofdE": "std::mem::align_of::<f64>",
            "_RNvCs1234_7mycrate3foo": "mycrate::foo",
            "_RNvNvCs1234_7mycrate3foo3bar": "mycrate::foo::bar",
            "_RNvNvCs1234_7mycrate4QUUX3FOO": "mycrate::QUUX::FOO",
            "_RNvNtNtCs1234_7mycrate3foo3bar3baz": "mycrate::foo::bar::baz",
            "_RNvC4core4iter": "core::iter",
            "_RNvNtC4core3fmt5write": "core::fmt::write",
            "_RNvNtNtC4core3fmt5write5write": "core::fmt::write::write",
            "_RNvC8closures5spawn": "closures::spawn",
            "_RNCINvC8closures5spawnE0": "closures::spawn::<>::{closure#0}",
            "_RNCINvCs1234_7mycrate3fooE3bar": "mycrate::foo::<>::{closure:bar#0}",
            "_RNCINvC3std4iterE0": "std::iter::<>::{closure#0}",
            "_RNCNCNgCs6DXkGYLi8lr_2cc5spawn00B5_": "cc::spawn::{closure#0}::{closure#0}",
            "_RNvNvMCs1234_7mycrateINtCs1234_7mycrate3FoopE3bar4QUUX": (
                "<mycrate::Foo<_>>::bar::QUUX"
            ),
            "_RNvNvXCs1234_7mycrateINtCs1234_7mycrate3FoopENtNtC3std5clone5Clone5clone4QUUX": (
                "<mycrate::Foo<_> as std::clone::Clone>::clone::QUUX"
            ),
            "_RINvNtC4core3mem4dropjE": "core::mem::drop::<usize>",
            "_RINvNtC4core3mem4swapRlRlE": "core::mem::swap::<&i32, &i32>",
            "_RINvNtC4core3mem4dropReE": "core::mem::drop::<&str>",
            "_RINvNtC4core3mem4dropRlE": "core::mem::drop::<&i32>",
            "_RINvNtC4core3mem4dropQlE": "core::mem::drop::<&mut i32>",
            "_RINvNtC4core3mem4dropPlE": "core::mem::drop::<*const i32>",
            "_RINvNtC4core3mem4dropOlE": "core::mem::drop::<*mut i32>",
            "_RINvNtC4core3mem4dropSlE": "core::mem::drop::<[i32]>",
            "_RINvNtC4core3mem4dropTllE": "core::mem::drop::<(i32, i32)>",
            "_RINvNtC4core3mem4dropTlE": "core::mem::drop::<(i32,)>",
        }
        for mangled, plain in cases.items():
            with self.subTest(mangled=mangled):
                self.assertEqual(demangle(mangled), plain)

    def test_rust_legacy_drops_hash(self) -> None:
        cases = {
            "_ZN3foo3bar17h0123456789abcdefE": "foo::bar",
            "_ZN3foo3barE": "foo::bar",
            "_ZN3foo3bar3bazE": "foo::bar::baz",
            "_ZN3foo3bar3baz3quxE": "foo::bar::baz::qux",
            "_ZN4core3ptr5writeE": "core::ptr::write",
            "_ZN4core3fmt5Debug3fmtE": "core::fmt::Debug::fmt",
            "_ZN4core3ops3Add3addE": "core::ops::Add::add",
            "_ZN5alloc3vec3Vec4pushE": "alloc::vec::Vec::push",
            "_ZN3std3vec3Vec3newE": "std::vec::Vec::new",
            "_ZN9my_module11my_functionE": "my_module::my_function",
            "_ZN9my_module11my_function17hfedcba9876543210E": "my_module::my_function",
            "__ZN3foo3barE": "foo::bar",
        }
        for mangled, plain in cases.items():
            with self.subTest(mangled=mangled):
                self.assertEqual(demangle(mangled), plain)

    def test_cxx_itanium(self) -> None:
        self.assertEqual(demangle("_ZN3hwy6N_AVX34FuncEv"), "hwy::N_AVX3::Func")

    def test_plain_c_symbol(self) -> None:
        self.assertEqual(demangle("kern_avx512"), "kern_avx512")
        self.assertEqual(demangle("__aarch64_ldadd4_relax"), "__aarch64_ldadd4_relax")


if __name__ == "__main__":
    unittest.main()
