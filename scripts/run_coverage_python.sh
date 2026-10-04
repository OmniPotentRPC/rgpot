#!/usr/bin/env bash
# Coverage for Python Cap'n Proto contract helpers under tests/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_XML="${1:-python_coverage.xml}"

python3 -m pip install -q pytest pytest-cov coverage 'pycapnp>=2' numpy

export PYTHONPATH="${ROOT}/tests${PYTHONPATH:+:$PYTHONPATH}"

COVRC=$(mktemp)
cat > "$COVRC" <<'CFG'
[run]
branch = True
source = tests
omit =
    tests/ase_*.py
    tests/con_*.py
    tests/eon_*.py
    tests/pes_*.py
    tests/rpc_integ.py
    tests/__pycache__/*
CFG
trap 'rm -f "$COVRC"' EXIT

PYTEST_ARGS=(
  tests/test_cpmd_params.py
  tests/test_nwchem_params.py
)
if [[ -n "${RGPOT_POTSERV:-}" && -x "${RGPOT_POTSERV}" ]]; then
  echo "    including optional potserv e2e tests"
  PYTEST_ARGS+=(
    tests/test_rpc_e2e_c_abi.py
    tests/test_rpc_integ_cpmd.py
    tests/test_rpc_integ_nwchem.py
  )
fi

echo "==> pytest python contract helpers"
python3 -m pytest "${PYTEST_ARGS[@]}" -q --tb=short \
  --cov=cpmd_params \
  --cov=nwchem_params \
  --cov-report=xml:"$ROOT/$OUT_XML" \
  --cov-report=term-missing \
  --cov-config="$COVRC"

test -s "$ROOT/$OUT_XML"
python3 - "$ROOT/$OUT_XML" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8", errors="replace")
print(f"python coverage XML size={p.stat().st_size}")
if "filename=" not in text:
    raise SystemExit("python coverage XML looks empty")
PY
echo "OK wrote $ROOT/$OUT_XML"
