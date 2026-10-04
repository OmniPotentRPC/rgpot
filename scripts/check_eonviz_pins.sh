#!/usr/bin/env bash
# Fail when an eonviz dependency is not an immutable pin.
#
# Rules for [feature.eonviz.pypi-dependencies]:
#   - a git dependency must name a full 40-character commit with `rev`
#     (no `branch`, no `tag`, no bare repository);
#   - a path or url dependency is rejected;
#   - chemparseplot and rgpycrumbs must be exact `==` release pins;
#   - every other entry must carry a version specifier (`*` is allowed only
#     for packages that are not part of the plotting stack).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
exec "$PY" - "$ROOT/pixi.toml" <<'PYEOF'
import re
import sys
import tomllib

STACK = {"chemparseplot", "rgpycrumbs"}
SHA = re.compile(r"^[0-9a-f]{40}$")
EXACT = re.compile(r"^==\s*\d[\w.+!-]*$")

with open(sys.argv[1], "rb") as fh:
    cfg = tomllib.load(fh)
deps = cfg["feature"]["eonviz"].get("pypi-dependencies", {})

bad = []
for name, spec in deps.items():
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

if bad:
    print("eonviz pin check failed:", file=sys.stderr)
    for line in bad:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
print(f"eonviz pin check ok ({len(deps)} pypi dependencies)")
PYEOF
