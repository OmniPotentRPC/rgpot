#!/usr/bin/env bash
# Install the eonviz readcon pin and compare atoms_to_con against the golden.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"
REV="$("$PY" - <<'PY'
import tomllib
spec = tomllib.load(open("pixi.toml", "rb"))["feature"]["eonviz"]["pypi-dependencies"]["readcon"]
if not isinstance(spec, dict) or "rev" not in spec:
    raise SystemExit("readcon pin is not a git rev; install that pin before running this check")
print(spec["rev"])
PY
)"
VENV="${READCON_GOLDEN_VENV:-/tmp/readcon-golden}"
uv venv "$VENV"
uv pip install --python "$VENV/bin/python" \
  pytest numpy ase \
  "readcon @ git+https://github.com/lode-org/readcon-core.git@${REV}"
"$VENV/bin/python" -m pytest -q tests/test_readcon_restart.py --tb=short
