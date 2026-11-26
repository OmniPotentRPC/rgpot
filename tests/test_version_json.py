# Copyright 2023--present rgpot developers
"""The citation files follow the release version and the CHANGELOG date.

cogapp regenerates the marked copies in ci/version/files.txt and passes a
listed file that has no markers. JSON cannot hold the markers, so
rgpot_version.py writes the JSON entries and check_drift.py compares them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci" / "version"))
import rgpot_version as v


def test_citation_files_are_listed():
    listed = v.listed_files()
    for name in ("CITATION.cff", "codemeta.json", ".zenodo.json"):
        assert name in listed


def test_every_non_json_entry_has_cog_markers():
    for name in v.listed_files():
        if name.endswith(".json"):
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "[[[cog" in text and "[[[end]]]" in text, name


def test_release_date_comes_from_the_changelog_heading():
    changelog = "## [9.1.0] - 2031-02-03\n\n## [9.0.0] - 2030-01-01\n"
    assert v.release_date("9.1.0", changelog) == "2031-02-03"
    assert v.release_date("9.0.0", changelog) == "2030-01-01"


def test_release_date_needs_the_heading():
    with pytest.raises(ValueError, match="towncrier build"):
        v.release_date("9.1.0", "## [9.0.0] - 2030-01-01\n")


def test_citation_cff_matches_the_release():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f'version: "{v.VERSION}"\n' in text
    assert f'date-released: "{v.release_date()}"\n' in text


def test_codemeta_uses_the_2_0_context():
    data = json.loads((ROOT / "codemeta.json").read_text(encoding="utf-8"))
    assert data["@context"] == "https://doi.org/10.5063/schema/codemeta-2.0"


def test_real_json_copies_are_in_step():
    assert v.stale_json() == []


def test_stale_json_is_reported_and_rewritten(tmp_path):
    doc = tmp_path / "codemeta.json"
    doc.write_text(
        json.dumps({"version": "0.0.1", "datePublished": "2000-01-01"}) + "\n",
        encoding="utf-8",
    )
    listing = tmp_path / "files.txt"
    listing.write_text("# comment\ncodemeta.json\n", encoding="utf-8")
    assert v.stale_json(root=tmp_path, listing=listing) == ["codemeta.json"]
    assert v.stale_json(write=True, root=tmp_path, listing=listing) == ["codemeta.json"]
    data = json.loads(doc.read_text(encoding="utf-8"))
    assert data == {"version": v.VERSION, "datePublished": v.release_date()}
    assert v.stale_json(root=tmp_path, listing=listing) == []
