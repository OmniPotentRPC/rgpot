# Copyright 2023--present rgpot developers
"""Coverage links may export libgcov. Other extra symbols still fail."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_engine_exports as exports


def test_gcov_runtime_names():
    assert exports.is_gcov_runtime("__gcov_init")
    assert exports.is_gcov_runtime("__gcov_merge_add")
    assert exports.is_gcov_runtime("mangle_path")
    assert exports.is_gcov_runtime("gcov_write")
    assert not exports.is_gcov_runtime("orcac_create")
    assert not exports.is_gcov_runtime("gcov_extra")


def test_coverage_build_keeps_real_leaks():
    expected = {"orcac_create", "orcac_destroy"}
    exported = expected | {
        "__gcov_init",
        "__gcov_var",
        "mangle_path",
        "gcov_write",
        "orcac_leaked",
    }
    assert exports.unexpected_symbols(exported, expected) == ["orcac_leaked"]


def test_non_coverage_still_reports_mangle_path():
    expected = {"orcac_create"}
    exported = {"orcac_create", "mangle_path", "gcov_write"}
    assert exports.unexpected_symbols(exported, expected) == [
        "gcov_write",
        "mangle_path",
    ]
