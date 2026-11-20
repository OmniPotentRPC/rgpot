"""Keep vendored vesin source and wheel constraints in lockstep."""

# Copyright 2023--present rgpot developers

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VESIN = ROOT / "third_party" / "vesin"
POTCTL_CI = ROOT / "potctl" / "src" / "ci.rs"
POTENTIALS_NCL = ROOT / "ci" / "gha" / "potentials.ncl"
RELEASE_RE = re.compile(r"v(\d+\.\d+\.\d+) plus local eOn patches\Z")
MARKER_RE = re.compile(r"^// vesin version: (\S+)\r?$", re.MULTILINE)
NCL_SPECS_RE = re.compile(r'"(vesin>=[^"]+)"\s+"(vesin-torch>=[^"]+)"')
# These read the release. They must not repeat the line from VERSION.
RELEASE_CONSUMERS = (
    ROOT / "meson.build",
    ROOT / "CMakeLists.txt",
    ROOT / "tests" / "python" / "test_vendored_vesin_version.py",
    ROOT / "potctl" / "src" / "ci.rs",
    ROOT / "docs" / "newsfragments" / "+vesin-vendored-ci.fixed.md",
)


def _release_lines() -> list[str]:
    return [
        line.strip()
        for line in (VESIN / "VERSION").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _read_vendored_release() -> str:
    lines = _release_lines()
    assert len(lines) == 1
    assert RELEASE_RE.fullmatch(lines[0]) is not None
    return lines[0]


def _release_pin(release: str) -> str:
    match = RELEASE_RE.fullmatch(release)
    assert match is not None
    return match.group(1)


def _marker_pin(text: str) -> str:
    match = MARKER_RE.search(text)
    assert match is not None
    return match.group(1)


def _read_ci_spec(name: str) -> str:
    text = POTCTL_CI.read_text(encoding="utf-8")
    match = re.search(rf'pub const {name}: &str = "([^"]+)";', text)
    assert match is not None
    return match.group(1)


def _assert_minor_series(spec: str, pin: str) -> None:
    major, minor, _patch = pin.split(".")
    lower = f">={major}.{minor}"
    upper = f"<{major}.{int(minor) + 1}"
    assert lower in spec
    assert upper in spec


def test_vendored_vesin_version_matches_source() -> None:
    pin = _release_pin(_read_vendored_release())
    header = (VESIN / "vesin-single-build.cpp").read_text(encoding="utf-8")
    assert pin == _marker_pin(header)


def test_vendored_vesin_marker_accepts_crlf_copy(tmp_path: Path) -> None:
    pin = _release_pin(_read_vendored_release())
    source = (VESIN / "vesin-single-build.cpp").read_bytes()
    lf = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    copy = tmp_path / "vesin-single-build.cpp"
    copy.write_bytes(crlf)
    text = copy.read_bytes().decode("utf-8")
    assert b"\r\n" in crlf
    assert b"\n" in crlf
    assert pin == _marker_pin(text)
    assert f"// vesin version: {pin}\n" in text.replace("\r\n", "\n")


def test_potctl_vesin_specs_match_vendored_minor_series() -> None:
    pin = _release_pin(_read_vendored_release())
    for name in ("VESIN_SPEC", "VESIN_TORCH_SPEC"):
        _assert_minor_series(_read_ci_spec(name), pin)


def test_potentials_ncl_vesin_specs_match_vendored_minor_series() -> None:
    pin = _release_pin(_read_vendored_release())
    text = POTENTIALS_NCL.read_text(encoding="utf-8")
    match = NCL_SPECS_RE.search(text)
    assert match is not None
    _assert_minor_series(match.group(1), pin)
    _assert_minor_series(match.group(2), pin)


def test_build_marker_checks_accept_crlf() -> None:
    for path in (ROOT / "meson.build", ROOT / "CMakeLists.txt"):
        text = path.read_text(encoding="utf-8")
        assert r"\r\n" in text, path


def test_release_is_defined_only_in_version() -> None:
    release = _read_vendored_release()
    for path in RELEASE_CONSUMERS:
        text = path.read_text(encoding="utf-8")
        assert release not in text, path
