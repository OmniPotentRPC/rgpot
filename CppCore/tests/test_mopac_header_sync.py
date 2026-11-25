#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Verify vendored mopac_c_abi.h matches the pinned mopacc commit."""

from __future__ import annotations

import argparse
import configparser
import difflib
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED_SHA256 = (
    "10fc452df7ab84dea2fd2ed2b5b6d1036f961000ca3e614358d89a971cf29de4"
)


def read_wrap(wrap_path: Path) -> tuple[str, str, str]:
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(wrap_path, encoding="utf-8")
    sec = cp["wrap-git"]
    return sec["url"].strip(), sec["revision"].strip(), sec["directory"].strip()


def fetch_via_git(url: str, revision: str) -> bytes | None:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "init", "-q"], cwd=tmp, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["git", "fetch", "-q", "--depth", "1", url, revision],
                cwd=tmp, check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            head = subprocess.run(
                ["git", "rev-parse", "FETCH_HEAD"],
                cwd=tmp, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True,
            ).stdout.strip()
            if head != revision:
                raise AssertionError(
                    f"fetched {head}, expected pinned {revision}"
                )
            return subprocess.run(
                ["git", "show", f"{revision}:include/mopacc.h"],
                cwd=tmp, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def fetch_via_https(url: str, revision: str) -> bytes | None:
    clean = url.removesuffix(".git").rstrip("/")
    if not clean.startswith("https://github.com/"):
        return None
    slug = clean.removeprefix("https://github.com/")
    raw_url = f"https://raw.githubusercontent.com/{slug}/{revision}/include/mopacc.h"
    try:
        with urllib.request.urlopen(raw_url, timeout=15) as resp:
            return resp.read()
    except (OSError, urllib.error.URLError):
        return None


def load_upstream(
    root: Path, url: str, revision: str, directory: str, require_fetch: bool
) -> tuple[bytes | None, str]:
    env_hdr = os.environ.get("RGPOT_MOPACC_HEADER", "").strip()
    if env_hdr:
        p = Path(env_hdr)
        return p.read_bytes(), str(p)

    env_dir = os.environ.get("RGPOT_MOPACC_DIR", "").strip()
    if env_dir:
        p = Path(env_dir) / "include" / "mopacc.h"
        return p.read_bytes(), str(p)

    for cand in (
        root / "subprojects" / directory / "include" / "mopacc.h",
        root / "subprojects" / "mopacc" / "include" / "mopacc.h",
    ):
        if cand.is_file():
            return cand.read_bytes(), str(cand)

    data = fetch_via_git(url, revision)
    if data is not None:
        return data, f"{url}@{revision}:include/mopacc.h"

    data = fetch_via_https(url, revision)
    if data is not None:
        return data, f"{url}@{revision}:include/mopacc.h"

    if require_fetch:
        raise AssertionError(
            f"unable to fetch include/mopacc.h from {url}@{revision}"
        )
    return None, "offline-digest-fallback"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    vendored_path = root / "CppCore" / "rgpot" / "MOPACPot" / "mopac_c_abi.h"
    wrap_path = root / "subprojects" / "mopacc.wrap"

    url, revision, directory = read_wrap(wrap_path)
    vendored = vendored_path.read_bytes()
    actual_sha = hashlib.sha256(vendored).hexdigest()

    upstream, source = load_upstream(
        root, url, revision, directory, args.fetch
    )
    if upstream is not None:
        if vendored != upstream:
            diff = "\n".join(
                difflib.unified_diff(
                    upstream.decode("utf-8", errors="replace").splitlines(),
                    vendored.decode("utf-8", errors="replace").splitlines(),
                    fromfile=source,
                    tofile=str(vendored_path.relative_to(root)),
                    lineterm="",
                )
            )
            print(diff, file=sys.stderr)
            raise AssertionError(
                "vendored mopac_c_abi.h diverges from pinned mopacc header"
            )
    if actual_sha != EXPECTED_SHA256:
        raise AssertionError(
            f"vendored mopac_c_abi.h sha256 {actual_sha} != {EXPECTED_SHA256}"
        )
    print(f"ok: mopac_c_abi.h matches {source} ({actual_sha[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
