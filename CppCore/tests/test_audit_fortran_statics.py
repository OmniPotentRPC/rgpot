#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Self-test for scripts/audit_fortran_statics.sh.

The audit must keep rejecting real procedure-local static storage (Fortran
SAVE variables and compiler-synthesized local static scratch) while accepting
documented profiling and coverage instrumentation symbols emitted by GCC,
Clang, Flang, and Intel ifx.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "scripts" / "audit_fortran_statics.sh"

CLEAN_F90 = textwrap.dedent(
    """\
    recursive subroutine clean_kernel(x, y)
      integer, intent(in) :: x
      integer, intent(out) :: y
      if (x > 0) then
        y = x + 1
      else
        y = x - 1
      end if
    end subroutine
    """
)

SAVE_F90 = textwrap.dedent(
    """\
    recursive subroutine leaky_kernel(x)
      integer, intent(out) :: x
      integer, save :: real_saved_hits = 0
      real_saved_hits = real_saved_hits + 1
      x = real_saved_hits
    end subroutine
    """
)

INSTRUMENTATION_ASM = textwrap.dedent(
    """\
      .bss
    __gcov0.sample_fn_:
      .zero 8
    __gcov7.sample_fn_:
      .zero 8
    __gcov0$sample_fn_:
      .zero 8
    __profc_sample_fn:
      .zero 8
    __profd_sample_fn:
      .zero 8
    __profn_sample_fn:
      .zero 8
    __profv_sample_fn:
      .zero 8
    __profvp_sample_fn:
      .zero 8
    __profbm_sample_fn:
      .zero 8
    __profvt_sample_fn:
      .zero 8
    __llvm_gcov_ctr:
      .zero 8
    __llvm_prf_vnodes:
      .zero 8
    __llvm_profile_runtime:
      .zero 8
    __llvm_coverage_mapping:
      .zero 8
    __covrec_deadbeef:
      .zero 8
    __sancov_gen_:
      .zero 8
      .data
    __gcov_.sample_fn_:
      .quad 0
    __gcov__sample_fn_:
      .quad 0
    __llvm_internal_gcov_emit_file_info:
      .quad 0
    """
)


def _find_gfortran() -> str | None:
    candidates = [os.environ.get("FC"), "gfortran"]
    for cand in candidates:
        if not cand:
            continue
        exe = shutil.which(cand)
        if not exe:
            continue
        ver = subprocess.run(
            [exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if ver.returncode == 0 and "GNU Fortran" in ver.stdout:
            return exe
    return None


class FortranStaticAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        if not sys.platform.startswith("linux"):
            self.skipTest("ELF nm audit targets Linux")
        if shutil.which("nm") is None or shutil.which("ar") is None or shutil.which("bash") is None:
            self.skipTest("nm, ar, or bash unavailable")
        holder = tempfile.TemporaryDirectory(prefix="rgpot-fortran-audit-")
        self.addCleanup(holder.cleanup)
        self.work = Path(holder.name)
        self.gfortran = _find_gfortran()
        self.assembler = shutil.which("as")

    def _run_audit(self, *targets: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(AUDIT), *(str(t) for t in targets)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _archive(self, name: str, *objects: Path) -> Path:
        archive = self.work / name
        subprocess.run(
            ["ar", "rcs", str(archive), *(str(o) for o in objects)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return archive

    def _nm_symbols(self, target: Path) -> str:
        return subprocess.run(
            ["nm", "-A", str(target)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout

    def test_real_save_is_rejected(self) -> None:
        if self.gfortran is None:
            self.skipTest("gfortran unavailable")
        src = self.work / "save.f90"
        obj = self.work / "save.o"
        src.write_text(SAVE_F90, encoding="utf-8")
        subprocess.run(
            [self.gfortran, "-frecursive", "-c", str(src), "-o", str(obj)],
            check=True,
        )
        archive = self._archive("libsave.a", obj)
        for target in (obj, archive):
            proc = self._run_audit(target)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("real_saved_hits", proc.stderr)

    def test_gcov_instrumented_clean_object_is_accepted(self) -> None:
        if self.gfortran is None:
            self.skipTest("gfortran unavailable")
        src = self.work / "clean.f90"
        obj = self.work / "clean_cov.o"
        src.write_text(CLEAN_F90, encoding="utf-8")
        subprocess.run(
            [
                self.gfortran,
                "-frecursive",
                "--coverage",
                "-fprofile-update=atomic",
                "-c",
                str(src),
                "-o",
                str(obj),
            ],
            check=True,
        )
        syms = self._nm_symbols(obj)
        self.assertIn("__gcov0.", syms)
        self.assertIn("__gcov_.", syms)
        archive = self._archive("libclean_cov.a", obj)
        for target in (obj, archive):
            proc = self._run_audit(target)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_gcov_instrumented_real_save_is_still_rejected(self) -> None:
        if self.gfortran is None:
            self.skipTest("gfortran unavailable")
        src = self.work / "save_cov.f90"
        obj = self.work / "save_cov.o"
        src.write_text(SAVE_F90, encoding="utf-8")
        subprocess.run(
            [
                self.gfortran,
                "-frecursive",
                "--coverage",
                "-fprofile-update=atomic",
                "-c",
                str(src),
                "-o",
                str(obj),
            ],
            check=True,
        )
        syms = self._nm_symbols(obj)
        self.assertIn("__gcov0.", syms)
        self.assertIn("real_saved_hits", syms)
        proc = self._run_audit(obj)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("real_saved_hits", proc.stderr)
        self.assertNotIn("__gcov", proc.stderr)

    def test_documented_llvm_and_gcov_prefixes_accepted_and_non_prefix_rejected(
        self,
    ) -> None:
        if self.assembler is None:
            self.skipTest("GNU assembler unavailable")
        asm_ok = self.work / "instr.s"
        obj_ok = self.work / "instr.o"
        asm_ok.write_text(INSTRUMENTATION_ASM, encoding="utf-8")
        subprocess.run([self.assembler, str(asm_ok), "-o", str(obj_ok)], check=True)
        proc_ok = self._run_audit(self._archive("libinstr.a", obj_ok))
        self.assertEqual(proc_ok.returncode, 0, proc_ok.stdout + proc_ok.stderr)

        asm_bad = self.work / "instr_bad.s"
        obj_bad = self.work / "instr_bad.o"
        asm_bad.write_text(
            INSTRUMENTATION_ASM + "  .bss\n_QMmodVreal_static:\n  .zero 4\n",
            encoding="utf-8",
        )
        subprocess.run([self.assembler, str(asm_bad), "-o", str(obj_bad)], check=True)
        proc_bad = self._run_audit(self._archive("libinstr_bad.a", obj_bad))
        self.assertEqual(proc_bad.returncode, 1, proc_bad.stdout + proc_bad.stderr)
        self.assertIn("_QMmodVreal_static", proc_bad.stderr)
        self.assertNotIn("__gcov", proc_bad.stderr)
        self.assertNotIn("__prof", proc_bad.stderr)
        self.assertNotIn("__llvm_", proc_bad.stderr)
        self.assertNotIn("__covrec_", proc_bad.stderr)
        self.assertNotIn("__sancov_", proc_bad.stderr)


if __name__ == "__main__":
    unittest.main()
