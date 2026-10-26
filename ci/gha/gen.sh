#!/usr/bin/env bash
# Export nickel-backed workflows that still ship as committed .yml (non-orchestrator).
# PR/main CI lives in hand-maintained .github/workflows/ci-orchestrator.yml (not exported).
# Job bodies for the orchestrator are mirrored in ci/gha/{build,prek,docs_quality,...}.ncl
# as documentation/audit sources (export-plan audit bundle), not as live workflows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

mapfile -t SPECS <<'EOF'
release.ncl|.github/workflows/release.yml
release_prepare.ncl|.github/workflows/release-prepare.yml
ci_docs.ncl|.github/workflows/ci_docs.yml
cosmo_potctl.ncl|.github/workflows/cosmo-potctl.yml
ci_doc_commenter.ncl|.github/workflows/ci_doc_commenter.yml
EOF

NICKEL=(nickel)
if ! command -v nickel >/dev/null 2>&1; then
  echo "nickel not on PATH; run via: pixi r -e cigen gen-gha" >&2
  exit 1
fi

for line in "${SPECS[@]}"; do
  src="${line%%|*}"
  dst="${line##*|}"
  echo "nickel export $src -> $dst"
  nickel export --format yaml "ci/gha/$src" -o "$dst"
  python3 ci/gha/annotate-pins.py "$dst"
done
echo "OK: ${#SPECS[@]} workflows exported (orchestrator is hand-maintained)"
