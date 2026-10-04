#!/usr/bin/env bash
# Regenerate nickel-exported workflows and fail if they differ from git.
# ci-orchestrator.yml is hand-maintained (not part of this gate).
# Entry points: pixi r -e cigen gha-drift   OR   bash ci/gha/check-drift.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if ! command -v nickel >/dev/null 2>&1; then
  echo "error: nickel not on PATH (use: pixi r -e cigen gha-drift)" >&2
  exit 1
fi

bash ci/gha/gen.sh
TRACKED=(
  .github/workflows/release.yml
  .github/workflows/release-prepare.yml
  .github/workflows/ci_docs.yml
  .github/workflows/cosmo-potctl.yml
  .github/workflows/ci_doc_commenter.yml
)
if ! git diff --exit-code -- "${TRACKED[@]}" >/dev/null 2>&1; then
  echo "error: nickel-exported workflows out of date vs ci/gha/*.ncl" >&2
  echo "fix: edit Nickel under ci/gha/, then: pixi r -e cigen gen-gha && git add .github/workflows" >&2
  echo "--- git diff --stat (tracked nickel exports) ---" >&2
  git diff --stat -- "${TRACKED[@]}" >&2 || true
  exit 1
fi
# Every third-party action ref is a full commit SHA with a trailing version comment.
if git grep -nE '^\s*(-\s*)?uses:\s*[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@' -- .github \
  | grep -vE 'uses:\s*\S+@[0-9a-f]{40} # \S+$'; then
  echo "error: action refs above are not pinned to a full SHA with a version comment" >&2
  exit 1
fi
echo "OK: nickel-exported workflows match ci/gha (orchestrator not in this gate)"
