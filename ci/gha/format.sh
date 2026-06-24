#!/usr/bin/env bash
# Format all Nickel sources under ci/gha/ (portable; no GNU xargs -r).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mapfile -t files < <(find ci/gha -name '*.ncl' | sort)
if ((${#files[@]} == 0)); then
  echo "no .ncl files under ci/gha"
  exit 0
fi
nickel format "${files[@]}"
echo "formatted ${#files[@]} file(s)"
