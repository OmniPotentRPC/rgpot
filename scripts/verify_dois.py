#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Check that every DOI the documentation cites exists.

The documentation cites papers as ``doi:10.xxxx/yyyy``.  A DOI "exists" when
the DOI handle service resolves it and Crossref or DataCite holds a record for
it.  The records worth keeping (title, first author, year, container) live in
``docs/dois.json`` so a later run can compare a citation against what the
registry said, offline.

Modes
-----
``--check`` (default, offline)
    Every DOI found in the documentation has an entry in the manifest, and the
    manifest holds no DOI the documentation does not cite.
``--update``
    Resolve the DOIs the manifest lacks (network) and write the manifest.
``--online``
    Resolve every DOI the documentation cites again and compare it with the
    manifest.
``DOI ...``
    Resolve the named DOIs, print each record, and add them to the manifest
    with ``--update``.

The module uses the standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "dois.json"
SOURCES = (
    "readme_src.org",
    "docs/*.md",
    "docs/*.org",
    "docs/orgmode/**/*.org",
)
USER_AGENT = "rgpot-docs (mailto:rgoswami@ieee.org)"
DOI_TOKEN = re.compile(r"10\.\d{4,9}/\S+")


def _strip(token: str) -> str:
    """Drop the punctuation a sentence puts after a DOI."""
    while token:
        last = token[-1]
        if last in ".,;:\"'=~*":
            token = token[:-1]
        elif last == ")" and token.count(")") > token.count("("):
            token = token[:-1]
        elif last == "]" and token.count("]") > token.count("["):
            token = token[:-1]
        else:
            break
    return token


def dois_in(text: str) -> set[str]:
    return {_strip(m.group(0)) for m in DOI_TOKEN.finditer(text)}


def cited_dois(root: Path = ROOT) -> dict[str, list[str]]:
    """DOI -> pages that cite it."""
    found: dict[str, list[str]] = {}
    seen: set[Path] = set()
    for pattern in SOURCES:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            rel = path.relative_to(root).as_posix()
            for doi in dois_in(path.read_text(encoding="utf-8")):
                found.setdefault(doi, []).append(rel)
    return found


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(data: dict[str, dict], path: Path = MANIFEST) -> None:
    ordered = {k: data[k] for k in sorted(data, key=str.lower)}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _get_json(url: str, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"no answer from {url}")


def handle_resolves(doi: str) -> bool:
    data = _get_json("https://doi.org/api/handles/" + urllib.parse.quote(doi, safe="/()"))
    return bool(data) and data.get("responseCode") == 1


def _crossref(doi: str) -> dict | None:
    data = _get_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/()"))
    if not data:
        return None
    msg = data["message"]
    authors = msg.get("author") or msg.get("editor") or []
    issued = (msg.get("issued") or {}).get("date-parts") or [[None]]
    return {
        "registry": "crossref",
        "title": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", msg["title"][0] if msg.get("title") else "")).strip(),
        "first_author": (authors[0].get("family") or authors[0].get("name", "")) if authors else "",
        "year": issued[0][0],
        "container": (msg.get("container-title") or [""])[0],
    }


def _datacite(doi: str) -> dict | None:
    data = _get_json("https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="/()"))
    if not data:
        return None
    attrs = data["data"]["attributes"]
    creators = attrs.get("creators") or []
    first = ""
    if creators:
        first = creators[0].get("familyName") or creators[0].get("name", "")
    return {
        "registry": "datacite",
        "title": (attrs.get("titles") or [{}])[0].get("title", ""),
        "first_author": first,
        "year": attrs.get("publicationYear"),
        "container": (attrs.get("container") or {}).get("title", "") or attrs.get("publisher") or "",
    }


def resolve(doi: str) -> dict | None:
    """Registry record of ``doi``, or None when the DOI does not resolve."""
    if not handle_resolves(doi):
        return None
    return _crossref(doi) or _datacite(doi)


def describe(doi: str, record: dict | None) -> str:
    if record is None:
        return f"{doi}: does not resolve"
    return (
        f"{doi}: {record['first_author']} ({record['year']}), "
        f"{record['container']}: {record['title']} [{record['registry']}]"
    )


def check(root: Path = ROOT) -> list[str]:
    cited = cited_dois(root)
    manifest = load_manifest(root / "docs" / "dois.json")
    problems = []
    for doi, pages in sorted(cited.items()):
        if doi not in manifest:
            problems.append(f"{pages[0]}: {doi} is not in docs/dois.json; run scripts/verify_dois.py --update")
    for doi in sorted(set(manifest) - set(cited)):
        problems.append(f"docs/dois.json: {doi} is cited nowhere; run scripts/verify_dois.py --update")
    return problems


def update(root: Path = ROOT, extra: list[str] | None = None) -> list[str]:
    cited = set(cited_dois(root)) | set(extra or [])
    manifest = load_manifest(root / "docs" / "dois.json")
    problems = []
    for doi in sorted(cited - set(manifest)):
        record = resolve(doi)
        print(describe(doi, record))
        if record is None:
            problems.append(f"{doi} does not resolve")
            continue
        manifest[doi] = record
    for doi in set(manifest) - cited:
        del manifest[doi]
    write_manifest(manifest, root / "docs" / "dois.json")
    return problems


def online(root: Path = ROOT) -> list[str]:
    cited = cited_dois(root)
    manifest = load_manifest(root / "docs" / "dois.json")
    problems = []
    for doi in sorted(cited):
        record = resolve(doi)
        if record is None:
            problems.append(f"{cited[doi][0]}: {doi} does not resolve")
            continue
        known = manifest.get(doi)
        if known and (known["title"], known["year"]) != (record["title"], record["year"]):
            problems.append(f"{doi}: the registry now says {describe(doi, record)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dois", nargs="*", help="resolve these DOIs and print the records")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="offline: manifest and documentation agree (default)")
    mode.add_argument("--update", action="store_true", help="resolve the missing DOIs and write the manifest")
    mode.add_argument("--online", action="store_true", help="resolve every cited DOI again")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    if args.dois and not args.update:
        bad = 0
        for doi in args.dois:
            record = resolve(doi)
            print(describe(doi, record))
            bad += record is None
        return 1 if bad else 0
    if args.update:
        problems = update(args.root, args.dois)
    elif args.online:
        problems = online(args.root)
    else:
        problems = check(args.root)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("verify_dois: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
