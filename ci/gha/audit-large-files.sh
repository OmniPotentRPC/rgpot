#!/usr/bin/env bash
# Fail if a tracked file exceeds the size threshold, except allowlisted paths.
# Allowlist matches prek.toml excludes (test goldens, vendored trees) plus the
# lockfile, which grows with dependencies. Threshold stays 1 MiB for the rest.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

THRESHOLD_BYTES="${THRESHOLD_BYTES:-$((1024 * 1024))}"

allowlisted() {
  case "$1" in
    CppCore/tests/data/* | third_party/* | llms-full.txt | pixi.lock) return 0 ;;
    *) return 1 ;;
  esac
}

fail=0
while IFS= read -r line; do
  meta="${line%%$'\t'*}"
  path="${line#*$'\t'}"
  size="${meta##* }"
  [[ -z "$path" || "$size" == "-" ]] && continue
  if allowlisted "$path"; then
    continue
  fi
  if ((size >= THRESHOLD_BYTES)); then
    printf '%s\t%s\n' "$size" "$path"
    fail=1
  fi
done < <(git ls-tree -r -l HEAD)

if ((fail)); then
  echo "error: tracked files at or above ${THRESHOLD_BYTES} bytes (allowlist: CppCore/tests/data/, third_party/, llms-full.txt, pixi.lock)" >&2
  exit 1
fi
echo "OK: no unexpected tracked file at or above ${THRESHOLD_BYTES} bytes"
