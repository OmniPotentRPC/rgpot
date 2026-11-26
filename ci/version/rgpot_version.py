"""Release version for cog snippets, read from the workspace Cargo.toml.

`[workspace.package] version` in the root Cargo.toml is the only place a
person writes the version. Every copy listed in ci/version/files.txt is
regenerated from it:

    pixi r -e version release-cog
    pixi r -e version version-drift

Call cogapp through `python -m cogapp`, because cocogitto also installs
a `cog` executable.

`python ci/version/rgpot_version.py header-macros [VERSION]` prints the
RGPOT_VERSION block. potctl's tests compare it with the block that
rgpot-core/version_macros.rs writes for the C header.

The release date is the one towncrier writes into the CHANGELOG.md
heading of VERSION. JSON has no comments to hold cog markers, so
`python ci/version/rgpot_version.py json [--check]` writes, or checks,
the version and date of the JSON entries in files.txt.
"""

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LISTING = ROOT / "ci" / "version" / "files.txt"

with (ROOT / "Cargo.toml").open("rb") as fh:
    VERSION = tomllib.load(fh)["workspace"]["package"]["version"]

_IDENT = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
_SEMVER = re.compile(
    r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    rf"(?:-{_IDENT}(?:\.{_IDENT})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)


def numeric_core(version):
    """MAJOR, MINOR and PATCH of a semver string, as integers."""
    match = _SEMVER.fullmatch(version)
    if match is None:
        raise ValueError(f"release version is not semver: {version!r}")
    return tuple(int(part) for part in match.groups())


MAJOR, MINOR, PATCH = numeric_core(VERSION)


def _dependency(features):
    if not features:
        return f'rgpot-core = "{VERSION}"'
    quoted = ", ".join(f'"{name}"' for name in features)
    return f'rgpot-core = {{ version = "{VERSION}", features = [{quoted}] }}'


def readme_md(*features):
    """A fenced TOML dependency block for rgpot-core/README.md."""
    return f"```toml\n[dependencies]\n{_dependency(features)}\n```"


def readme_org(*features):
    """A TOML source block for rgpot-core/README.org."""
    return f"#+begin_src toml\n[dependencies]\n{_dependency(features)}\n#+end_src"


def header_macros(version=VERSION):
    """The RGPOT_VERSION block that rgpot-core/build.rs gives cbindgen."""
    major, minor, patch = numeric_core(version)
    return "\n".join(
        [
            f'#define RGPOT_VERSION "{version}"',
            f"#define RGPOT_VERSION_MAJOR {major}",
            f"#define RGPOT_VERSION_MINOR {minor}",
            f"#define RGPOT_VERSION_PATCH {patch}",
        ]
    )


def release_date(version=VERSION, changelog=None):
    """The YYYY-MM-DD date of the CHANGELOG.md heading for version."""
    if changelog is None:
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$", re.M
    )
    match = heading.search(changelog)
    if match is None:
        raise ValueError(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' heading; "
            "run towncrier build before release-cog"
        )
    return match.group(1)


_JSON_DATE_KEYS = ("datePublished", "publication_date")


def listed_files(listing=LISTING):
    """Entries of ci/version/files.txt, relative to the repository root."""
    lines = listing.read_text(encoding="utf-8").splitlines()
    return [s for s in (line.strip() for line in lines) if s and not s.startswith("#")]


def json_with_release(text, version=VERSION, date=None):
    """The JSON document with its version and release date replaced."""
    data = json.loads(text)
    data["version"] = version
    for key in _JSON_DATE_KEYS:
        if key in data:
            data[key] = date if date is not None else release_date(version)
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def stale_json(write=False, root=ROOT, listing=LISTING):
    """JSON entries of files.txt whose version or date differs; optionally rewrite them."""
    stale = []
    date = release_date()
    for name in listed_files(listing):
        if not name.endswith(".json"):
            continue
        path = root / name
        text = path.read_text(encoding="utf-8")
        new = json_with_release(text, VERSION, date)
        if new != text:
            stale.append(name)
            if write:
                path.write_text(new, encoding="utf-8")
    return stale


if __name__ == "__main__":
    if len(sys.argv) in (2, 3) and sys.argv[1] == "header-macros":
        print(header_macros(*sys.argv[2:]))
    elif sys.argv[1:] == ["json"]:
        for name in stale_json(write=True):
            print(f"Updated {name}")
    elif sys.argv[1:] == ["json", "--check"]:
        stale = stale_json()
        if stale:
            sys.exit("version or release date drifted in: " + ", ".join(stale))
    else:
        sys.exit(
            f"usage: {sys.argv[0]} header-macros [VERSION] | json [--check]"
        )
