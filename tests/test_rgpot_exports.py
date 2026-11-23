# Copyright 2023--present rgpot developers
"""The shared library's rgpot_* export count stays at the measured pin.

51 for a normal link of librgpot_core.a, 105 when librgpot_bundle.a is
linked with --whole-archive. The extra 54 are the public C ABI in the
core baseline plus the four fast-list entry points. Two of the umbrella
exports are the Muller-Brown entry points.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_rgpot_exports as exports


def _baseline(name: str) -> set[str]:
    return exports.read_baseline(ROOT / "scripts" / name)


def test_pinned_counts():
    assert exports.CORE_ONLY_EXPORTS == 51
    assert exports.BUNDLE_EXPORTS == 105
    assert exports.BUNDLE_EXPORTS - exports.CORE_ONLY_EXPORTS == 54


def test_baselines_are_the_public_archive_set():
    core = _baseline("abi_core_baseline.txt")
    fast = _baseline("abi_baseline_fast.txt")
    assert len(core) == 50
    assert len(fast) == 4
    assert core.isdisjoint(fast)
    assert len(core | fast) == 54


def test_whole_archive_exports_the_public_set():
    extra = exports.BUNDLE_EXPORTS - exports.CORE_ONLY_EXPORTS
    wanted = {f"rgpot_abi_{index}" for index in range(extra)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella | wanted, wanted, wanted, whole=True)


def test_core_only_exports_none_of_the_archive():
    wanted = {f"rgpot_abi_{index}" for index in range(42)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella, wanted, wanted, whole=False)


def test_core_only_rejects_an_archive_symbol():
    wanted = {f"rgpot_abi_{index}" for index in range(42)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    umbrella.add("rgpot_abi_0")
    with pytest.raises(SystemExit, match="exported rust archive symbols"):
        exports.evaluate(umbrella, wanted, wanted, whole=False)


def test_whole_archive_rejects_a_dropped_name():
    extra = exports.BUNDLE_EXPORTS - exports.CORE_ONLY_EXPORTS
    wanted = {f"rgpot_abi_{index}" for index in range(extra)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exported = (umbrella | wanted) - {"rgpot_abi_0"}
    with pytest.raises(SystemExit, match="dropped public"):
        exports.evaluate(exported, wanted, wanted, whole=True)


def test_count_mismatch_names_the_pin():
    extra = exports.BUNDLE_EXPORTS - exports.CORE_ONLY_EXPORTS
    wanted = {f"rgpot_abi_{index}" for index in range(extra)}
    exported = {f"rgpot_cxx_{index}" for index in range(exports.BUNDLE_EXPORTS)}
    exported.update(wanted)
    with pytest.raises(SystemExit, match="expected 105"):
        exports.evaluate(exported, wanted, wanted, whole=True)


def test_nm_listing_keeps_defined_rgpot_names():
    text = "\n".join(
        [
            "0000000000000000 T rgpot_version",
            "                 U rgpot_potential_new",
            "0000000000000000 T not_rgpot",
            "0000000000000000 R rgpot_abi_stamp",
        ]
    )
    assert exports.rgpot_names(text) == {"rgpot_version", "rgpot_abi_stamp"}


def test_session_and_combined_counts_are_pinned():
    assert exports.SESSION_EXPORTS == 21
    assert exports.CORE_ONLY_COMBINED_EXPORTS == 72
    assert exports.BUNDLE_COMBINED_EXPORTS == 126


def test_rpc_baseline_is_an_optional_archive_set():
    base = _baseline("abi_core_baseline.txt") | _baseline("abi_baseline_fast.txt")
    rpc = _baseline("abi_baseline_rpc.txt")
    assert len(rpc) == 7
    assert base.isdisjoint(rpc)


def test_whole_archive_count_tracks_optional_public_exports():
    wanted = {f"rgpot_abi_{index}" for index in range(61)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella | wanted, wanted, wanted, whole=True)


def test_session_baseline_is_its_own_pin():
    session = exports.read_baseline(exports.SESSION_BASELINE)
    assert len(session) == 21
    assert all(name.startswith("rgpot_md_session_") for name in session)
    assert session.isdisjoint(_baseline("abi_core_baseline.txt"))
    assert session.isdisjoint(_baseline("abi_baseline_fast.txt"))


def test_session_names_are_not_part_of_the_counts():
    session = {f"rgpot_md_session_{index}" for index in range(21)}
    wanted = {f"rgpot_abi_{index}" for index in range(5)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exports.evaluate(umbrella | session, wanted, wanted, whole=False, session=session)
    with pytest.raises(SystemExit, match="expected 51"):
        exports.evaluate(umbrella | session, wanted, wanted, whole=False)


def test_a_lost_session_export_is_named():
    session = {f"rgpot_md_session_{index}" for index in range(21)}
    wanted = {f"rgpot_abi_{index}" for index in range(5)}
    umbrella = {f"rgpot_cxx_{index}" for index in range(exports.CORE_ONLY_EXPORTS)}
    exported = (umbrella | session) - {"rgpot_md_session_3"}
    with pytest.raises(SystemExit, match="rgpot_md_session_3"):
        exports.evaluate(exported, wanted, wanted, whole=False, session=session)
