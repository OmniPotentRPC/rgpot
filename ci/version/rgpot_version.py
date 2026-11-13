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
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

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


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or sys.argv[1] != "header-macros":
        sys.exit(f"usage: {sys.argv[0]} header-macros [VERSION]")
    print(header_macros(*sys.argv[2:]))
