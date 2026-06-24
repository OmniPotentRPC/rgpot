#!/usr/bin/env bash
# Export every ci/gha/*.ncl workflow (except lib/) into .github/workflows/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

mapfile -t SPECS <<'EOF'
build.ncl|.github/workflows/build.yml
potentials.ncl|.github/workflows/potentials.yml
release.ncl|.github/workflows/release.yml
release_prepare.ncl|.github/workflows/release-prepare.yml
prek.ncl|.github/workflows/prek.yml
ci_docs.ncl|.github/workflows/ci_docs.yml
docs_quality.ncl|.github/workflows/docs_quality.yml
towncrier.ncl|.github/workflows/towncrier.yml
cosmo_potctl.ncl|.github/workflows/cosmo-potctl.yml
cosmo_potctl_spike.ncl|.github/workflows/cosmo-potctl-spike.yml
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
done
echo "OK: ${#SPECS[@]} workflows exported"
