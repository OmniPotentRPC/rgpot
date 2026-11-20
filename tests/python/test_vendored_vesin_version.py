"""Keep vendored vesin source and wheel constraints in lockstep."""

# Copyright 2023--present rgpot developers

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VESIN = ROOT / "third_party" / "vesin"
POTCTL_CI = ROOT / "potctl" / "src" / "ci.rs"
VENDORED_RELEASE = "v0.6.0 plus local eOn patches"


def _release_lines() -> list[str]:
    return [
        line.strip()
        for line in (VESIN / "VERSION").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_vendored_release() -> str:
    lines = _release_lines()
    assert lines == [VENDORED_RELEASE]
    return lines[0]


def _release_pin(release: str) -> str:
    match = re.fullmatch(r"v(\d+\.\d+\.\d+) plus local eOn patches", release)
    assert match is not None
    return match.group(1)


def _read_ci_spec(name: str) -> str:
    text = POTCTL_CI.read_text()
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', text)
    assert match is not None
    return match.group(1)


def test_vendored_vesin_version_matches_source() -> None:
    pin = _release_pin(_read_vendored_release())
    header = (VESIN / "vesin-single-build.cpp").read_text()
    match = re.search(r"^// vesin version: (\S+)$", header, re.MULTILINE)
    assert match is not None
    assert pin == match.group(1)


def test_potctl_vesin_specs_match_vendored_minor_series() -> None:
    pin = _release_pin(_read_vendored_release())
    major, minor, *_ = pin.split(".")
    next_minor = f"{major}.{int(minor) + 1}"
    for name in ("VESIN_SPEC", "VESIN_TORCH_SPEC"):
        spec = _read_ci_spec(name)
        assert f">={major}.{minor}" in spec
        assert f"<{next_minor}" in spec
