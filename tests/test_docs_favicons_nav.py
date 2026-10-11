"""Docs chrome: favicon files exist, and the Ecosystem menu is the shared list."""

import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAVICONS = ROOT / "docs" / "source" / "_static" / "favicons"
CONF = ROOT / "docs" / "source" / "conf.py"

# Shared Ecosystem list. Each site drops its own row.
ECOSYSTEM = [
    ("eOn", "https://eondocs.org"),
    ("rgpot", "https://rgpot.rgoswami.me"),
    ("rgmin", "https://omnipotentrpc.github.io/rgmin"),
    ("rgsaddle", "https://omnipotentrpc.github.io/rgsaddle"),
    ("cpmdc", "https://cpmdc.rgoswami.me"),
    ("rgpycrumbs", "https://rgpycrumbs.rgoswami.me"),
    ("minimage", "https://lode-org.github.io/minimage"),
    ("linkcell", "https://docs.rs/linkcell"),
    ("readcon", "https://lode-org.github.io/readcon-core"),
    ("readcon-db", "https://lode-org.github.io/readcon-db"),
    ("readcon-ops", "https://github.com/lode-org/readcon-ops"),
]

PNG_SIZES = {
    "favicon-16x16.png": (16, 16),
    "favicon-32x32.png": (32, 32),
    "apple-touch-icon.png": (180, 180),
    "android-chrome-192x192.png": (192, 192),
    "android-chrome-512x512.png": (512, 512),
}


def _png_size(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


def test_favicon_set_matches_extra_head_links():
    extra = (ROOT / "docs/source/_templates/partials/extra-head.html").read_text()
    for name, size in PNG_SIZES.items():
        assert name in extra
        path = FAVICONS / name
        assert path.is_file() and path.stat().st_size > 0
        assert _png_size(path) == size
    svg = (FAVICONS / "favicon.svg").read_bytes()
    assert svg.startswith(b"<svg")
    assert b"base64" not in svg
    ico = (FAVICONS / "favicon.ico").read_bytes()
    assert ico[:4] == b"\x00\x00\x01\x00"
    count = struct.unpack_from("<H", ico, 4)[0]
    assert count >= 3
    assert "apple-touch-icon" in extra
    assert "site.webmanifest" in extra
    manifest = (FAVICONS / "site.webmanifest").read_text()
    assert "android-chrome-192x192.png" in manifest
    assert "android-chrome-512x512.png" in manifest
    conf = CONF.read_text()
    assert 'html_favicon = "_static/favicons/favicon.svg"' in conf
    assert '"light_logo": "_static/logo-light.svg"' in conf
    assert '"dark_logo": "_static/logo-dark.svg"' in conf


def test_v2_branding_keeps_old_files_and_vector_mascot():
    logo = ROOT / "branding" / "logo"
    for name in (
        "rgpot_logo.webp",
        "rgpot_logo_light.webp",
        "rgpot_logo_dark.webp",
        "rgpot_notext_light.webp",
        "rgpot_notext_dark.webp",
        "logo-light.svg",
        "logo-dark.svg",
        "mascot-light.svg",
        "mascot-dark.svg",
        "mascot-light.webp",
        "mascot-dark.webp",
        "icon.svg",
        "favicon/favicon.svg",
    ):
        assert (logo / name).is_file()
    notext = (logo / "rgpot_notext.svg").read_bytes()
    assert notext == (logo / "mascot-light.svg").read_bytes()
    assert b"base64" not in notext
    assert notext.startswith(b"<svg")
    static = ROOT / "docs" / "source" / "_static"
    assert (static / "logo-light.svg").read_bytes() == (logo / "logo-light.svg").read_bytes()
    assert (static / "logo-dark.svg").read_bytes() == (logo / "logo-dark.svg").read_bytes()
    assert (FAVICONS / "favicon.svg").read_bytes() == (logo / "favicon" / "favicon.svg").read_bytes()


def test_explanation_section_is_org_source():
    index = (ROOT / "docs/orgmode/index.org").read_text()
    assert ":caption: Explanation" in index
    assert "explanation/index" in index
    assert (ROOT / "docs/orgmode/explanation/index.org").is_file()
    assert (ROOT / "docs/orgmode/explanation/rpc-model.org").is_file()
    readme = (ROOT / "readme_src.org").read_text()
    assert "https://rgpot.rgoswami.me" in readme


def test_ecosystem_nav_is_the_shared_list():
    text = CONF.read_text()
    positions = []
    for title, url in ECOSYSTEM:
        assert url in text
        positions.append(text.index(url))
        assert title in text
    assert positions == sorted(positions)
    assert "Python bindings for rgpot" not in text
    assert "https://github.com/HaoZeke/readcon-db" not in text
