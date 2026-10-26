#!/usr/bin/env bash
# Fail when an eonviz dependency is not an immutable pin.
#
# Rules for [feature.eonviz.pypi-dependencies]:
#   - a git dependency must name a full 40-character commit with `rev`
#     (no `branch`, no `tag`, no bare repository);
#   - a path or url dependency is rejected;
#   - chemparseplot and rgpycrumbs must be exact `==` release pins;
#   - readcon's minimum is parsed from the dependency. A version string
#     must be `>=0.17.1` or a higher floor (or an exact `==` at least that
#     high). A git pin, used until the 0.17.1 wheel is published, must be
#     readcon-core 694e808 or a descendant, and the script parses `version`
#     from that commit's pyproject.toml. That parsed version must be
#     >=0.17.1. A floor the script cannot parse fails the job.
#   - every other entry must carry a version specifier (`*` is allowed only
#     for packages that are not part of the plotting stack).
#   - `format="eon"` outside a comment fails the job.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
TOML="${EONVIZ_TOML:-$ROOT/pixi.toml}"
exec "$PY" - "$TOML" "$ROOT" <<'PYEOF'
import re
import sys
import tomllib
from pathlib import Path

STACK = {"chemparseplot", "rgpycrumbs"}
SHA = re.compile(r"^[0-9a-f]{40}$")
EXACT = re.compile(r"^==\s*\d[\w.+!-]*$")
FLOOR = re.compile(r">=\s*(\d+)\.(\d+)\.(\d+)")
EXACT_VER = re.compile(r"^==\s*(\d+)\.(\d+)\.(\d+)")
PROJECT_VERSION = re.compile(r'(?m)^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"')
READCON_MIN = (0, 17, 1)
READCON_GIT_FLOOR = "694e808080c49979ee3099afcfa18e5420c40433"
EON_FORMAT = "format=" + '"eon"'
EON_FORMAT_SQ = "format=" + "'eon'"
SKIP_DIRS = {
    ".git",
    ".pixi",
    "__pycache__",
    "target",
    "node_modules",
    "bbdir",
    ".cosmo-build",
}


def parsed_triple(match) -> tuple[int, int, int] | None:
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


def parsed_floor(spec: str):
    text = spec.strip()
    exact = EXACT_VER.match(text)
    if exact is not None:
        return parsed_triple(exact)
    return parsed_triple(FLOOR.search(text))


def version_at_rev(rev: str) -> tuple[int, int, int] | None:
    import urllib.request

    url = f"https://raw.githubusercontent.com/lode-org/readcon-core/{rev}/pyproject.toml"
    request = urllib.request.Request(url, headers={"User-Agent": "rgpot-eonviz-pins"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode()
    return parsed_triple(PROJECT_VERSION.search(text))


def readcon_floor_error(spec) -> str | None:
    version = None
    rev = None
    if isinstance(spec, dict):
        if "branch" in spec or "tag" in spec:
            return "readcon: git dependency uses branch/tag, pin `rev`"
        if "git" in spec:
            rev = str(spec.get("rev", ""))
            if not SHA.match(rev):
                return "readcon: git dependency needs a 40-hex `rev`"
        elif "path" in spec or "url" in spec:
            return "readcon: path/url dependencies are not immutable pins"
        if "version" in spec:
            version = spec["version"]
    elif isinstance(spec, str):
        version = spec
    else:
        return f"readcon: unrecognised dependency {spec!r}"
    if rev is not None:
        if rev != READCON_GIT_FLOOR and not _rev_descends_from_floor(rev):
            return (
                "readcon: git rev must be "
                f"{READCON_GIT_FLOOR} or a descendant, got {rev}"
            )
        try:
            got = version_at_rev(rev)
        except Exception as exc:
            return f"readcon: could not parse version at {rev}: {exc}"
        if got is None or got < READCON_MIN:
            return (
                "readcon: parsed version "
                f"{got} at {rev} is below >=0.17.1"
            )
    if rev is None and not isinstance(version, str):
        return "readcon: minimum >=0.17.1 is not in the dependency the script parses"
    if isinstance(version, str):
        floor = parsed_floor(version)
        if floor is None or floor < READCON_MIN:
            return f"readcon: parsed floor {floor} is below >=0.17.1, got {version!r}"
    return None


def _rev_descends_from_floor(rev: str) -> bool:
    import json
    import urllib.request

    url = (
        "https://api.github.com/repos/lode-org/readcon-core/compare/"
        f"{READCON_GIT_FLOOR}...{rev}"
    )
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload.get("status") in {"ahead", "identical"}


def code_without_comment(line: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(line):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#":
            return line[:index]
    return line


def eon_format_hits(root: Path) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            code = code_without_comment(line)
            if EON_FORMAT in code or EON_FORMAT_SQ in code:
                hits.append(f"{path.relative_to(root)}:{number}")
    return hits


toml_path, root = sys.argv[1], Path(sys.argv[2])
with open(toml_path, "rb") as fh:
    cfg = tomllib.load(fh)
deps = cfg["feature"]["eonviz"].get("pypi-dependencies", {})

bad = []
for name, spec in deps.items():
    if name == "readcon":
        message = readcon_floor_error(spec)
        if message:
            bad.append(message)
        continue
    if isinstance(spec, dict):
        if "git" in spec:
            if "branch" in spec or "tag" in spec:
                bad.append(f"{name}: git dependency uses branch/tag, pin `rev`")
            elif not SHA.match(str(spec.get("rev", ""))):
                bad.append(f"{name}: git dependency needs a 40-hex `rev`")
        elif "path" in spec or "url" in spec:
            bad.append(f"{name}: path/url dependencies are not immutable pins")
        elif "version" in spec:
            spec = spec["version"]
        else:
            bad.append(f"{name}: unrecognised dependency table")
            continue
    if isinstance(spec, str):
        if name in STACK and not EXACT.match(spec):
            bad.append(f"{name}: must be an exact `==` release pin, got {spec!r}")
        elif spec.strip() in ("", "*") and name in STACK:
            bad.append(f"{name}: unpinned")

hits = eon_format_hits(root)
for hit in hits:
    bad.append(f"format=eon outside a comment: {hit}")

if bad:
    print("eonviz pin check failed:", file=sys.stderr)
    for line in bad:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
print(f"eonviz pin check ok ({len(deps)} pypi dependencies)")
PYEOF
