#!/usr/bin/env bash
# Regenerate GHA YAML from Nickel and fail if tracked workflows differ.
# Entry points: pixi r -e cigen gha-drift   OR   bash ci/gha/check-drift.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v nickel >/dev/null 2>&1; then
  echo "error: nickel not on PATH (use: pixi r -e cigen gha-drift)" >&2
  exit 1
fi

bash ci/gha/gen.sh
if ! git diff --exit-code -- .github/workflows >/dev/null 2>&1; then
  echo "error: .github/workflows is out of date vs ci/gha/*.ncl" >&2
  echo "fix: edit Nickel under ci/gha/, then: pixi r -e cigen gen-gha && git add .github/workflows" >&2
  echo "--- git diff --stat .github/workflows ---" >&2
  git diff --stat -- .github/workflows >&2 || true
  exit 1
fi
echo "OK: .github/workflows matches ci/gha (no drift)"
