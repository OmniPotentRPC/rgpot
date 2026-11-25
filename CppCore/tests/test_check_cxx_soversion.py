#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Self-test for the Neighbours generation-symbol check in check_cxx_soversion.py.

The check picks the nm invocation from each file's magic bytes: ELF reads
the dynamic table, Mach-O the external defined symbols, and anything else
(a Windows DLL) is skipped. These cases write stand-in files with each
magic and replace subprocess.run, so the Mach-O path runs on any host. One
case links a real ELF shared object when a C++ compiler and nm are present.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_cxx_soversion as ccs  # noqa: E402

LINUX_NM = """\
0000000000001000 T rgpot::nlist::Neighbours::nextGeneration()
0000000000001010 T rgpot::nlist::Neighbours::assignGeneration() const
0000000000001020 T rgpot::nlist::Neighbours::exchangeGenerationCounter(unsigned long)
0000000000001030 T rgpot_abi_version
"""

# llvm-nm -g -U -C on a universal dylib: each slice repeats the symbols,
# and std::uint64_t demangles as unsigned long long.
DARWIN_FAT_NM = """\

librgpot.9.dylib (for architecture x86_64):
0000000000001000 T rgpot::nlist::Neighbours::nextGeneration()
0000000000001010 T rgpot::nlist::Neighbours::assignGeneration() const
0000000000001020 T rgpot::nlist::Neighbours::exchangeGenerationCounter(unsigned long long)

librgpot.9.dylib (for architecture arm64):
0000000000001000 T rgpot::nlist::Neighbours::nextGeneration()
0000000000001010 T rgpot::nlist::Neighbours::assignGeneration() const
0000000000001020 T rgpot::nlist::Neighbours::exchangeGenerationCounter(unsigned long long)
"""

MACHO_64_LE = bytes.fromhex("cffaedfe")
FAT = bytes.fromhex("cafebabe")


class FakeNm:
    """subprocess.run stand-in that answers nm by file name."""

    def __init__(self, outputs: dict[str, str]) -> None:
        self.outputs = outputs
        self.commands: list[list[str]] = []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        stdout = self.outputs[Path(command[-1]).name]
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


class GenerationSymbolDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write(self, name: str, magic: bytes) -> Path:
        path = self.root / name
        path.write_bytes(magic + b"\0" * 60)
        return path

    def run_check(self, lib: Path, engines: list[Path], nm: FakeNm) -> str:
        out = io.StringIO()
        with mock.patch.object(ccs.subprocess, "run", nm), redirect_stdout(out):
            ccs.check_generation_symbol(lib, engines)
        return out.getvalue()

    def test_binary_format_reads_the_magic(self) -> None:
        self.assertEqual(ccs.binary_format(self.write("a.so", b"\x7fELF")), "elf")
        for word in ("feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe"):
            path = self.write(f"{word}.dylib", bytes.fromhex(word))
            self.assertEqual(ccs.binary_format(path), "macho", word)
        self.assertIsNone(ccs.binary_format(self.write("a.dll", b"MZ\x90\0")))
        self.assertIsNone(ccs.binary_format(self.write("empty", b"")))

    def test_nm_command_follows_the_format(self) -> None:
        lib = Path("librgpot.so")
        self.assertEqual(
            ccs.nm_command(lib, "elf"),
            ["nm", "-D", "--defined-only", "--demangle", "librgpot.so"],
        )
        self.assertEqual(
            ccs.nm_command(lib, "macho"), ["nm", "-g", "-U", "-C", "librgpot.so"]
        )

    def test_macho_umbrella_and_plugins_are_inspected(self) -> None:
        lib = self.write("librgpot.9.dylib", FAT)
        engine = self.write("liblj_engine.dylib", MACHO_64_LE)
        nm = FakeNm({lib.name: DARWIN_FAT_NM, engine.name: ""})
        output = self.run_check(lib, [engine], nm)
        self.assertEqual(
            nm.commands,
            [ccs.nm_command(lib, "macho"), ccs.nm_command(engine, "macho")],
        )
        self.assertIn("three exported Neighbours generation symbols", output)
        self.assertIn("in 1 engines", output)

    def test_macho_plugin_with_private_counter_fails(self) -> None:
        lib = self.write("librgpot.9.dylib", MACHO_64_LE)
        engine = self.write("libnative_lj_engine.dylib", MACHO_64_LE)
        nm = FakeNm({lib.name: DARWIN_FAT_NM, engine.name: DARWIN_FAT_NM})
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_check(lib, [engine], nm)

    def test_elf_umbrella_with_mixed_plugins(self) -> None:
        lib = self.write("librgpot.so.9", ccs.ELF_MAGIC)
        elf_engine = self.write("liblj_engine.so", ccs.ELF_MAGIC)
        dll_engine = self.write("lj_engine.dll", b"MZ\x90\0")
        archive_engine = self.write("liblj_engine.a", b"!<ar")
        nm = FakeNm({lib.name: LINUX_NM, elf_engine.name: ""})
        output = self.run_check(lib, [elf_engine, dll_engine, archive_engine], nm)
        self.assertEqual(
            nm.commands,
            [ccs.nm_command(lib, "elf"), ccs.nm_command(elf_engine, "elf")],
        )
        self.assertIn("in 1 engines", output)

    def test_static_umbrella_still_checks_plugins(self) -> None:
        lib = self.write("librgpot.a", b"!<ar")
        engine = self.write("liblj_engine.so", ccs.ELF_MAGIC)
        nm = FakeNm({engine.name: LINUX_NM})
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_check(lib, [engine], nm)
        self.assertEqual(nm.commands, [ccs.nm_command(engine, "elf")])

    def test_windows_umbrella_is_skipped(self) -> None:
        lib = self.write("rgpot.dll", b"MZ\x90\0")
        nm = FakeNm({})
        self.run_check(lib, [self.write("lj_engine.dll", b"MZ\x90\0")], nm)
        self.assertEqual(nm.commands, [])

    def test_missing_umbrella_symbol_fails(self) -> None:
        lib = self.write("librgpot.so.9", ccs.ELF_MAGIC)
        nm = FakeNm({lib.name: LINUX_NM.splitlines()[0] + "\n"})
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            self.run_check(lib, [], nm)


CXX = os.environ.get("CXX", "c++")


@unittest.skipUnless(
    sys.platform.startswith("linux") and shutil.which(CXX) and shutil.which("nm"),
    "needs a Linux C++ compiler and nm",
)
class RealElfTests(unittest.TestCase):
    def test_linked_shared_object(self) -> None:
        source = """
            #include <cstdint>
            namespace rgpot::nlist {
            struct Neighbours {
              static std::uint64_t nextGeneration();
              std::uint64_t assignGeneration() const;
              static std::uint64_t exchangeGenerationCounter(std::uint64_t) noexcept;
            };
            std::uint64_t Neighbours::nextGeneration() { return 1; }
            std::uint64_t Neighbours::assignGeneration() const { return 2; }
            std::uint64_t Neighbours::exchangeGenerationCounter(std::uint64_t v) noexcept {
              return v;
            }
            }
        """
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "gen.cc"
            src.write_text(source)
            lib = Path(tmp) / "libgen.so"
            link = subprocess.run(
                [CXX, "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(lib)],
                capture_output=True,
                text=True,
            )
            if link.returncode != 0:
                self.skipTest(f"{CXX} cannot link a shared object: {link.stderr}")
            self.assertEqual(ccs.binary_format(lib), "elf")
            self.assertEqual(len(ccs.defined_generation_symbols(lib, "elf")), 3)


if __name__ == "__main__":
    unittest.main()
