#!/usr/bin/env bash
# Install apt packages, retrying transient mirror and DNS failures.
# Exit 2 when a package name does not exist, so the caller can fall back
# without waiting out the retries. Exit 1 when the mirror never answers.
set -euo pipefail

extra=()
if [ "${1:-}" = "--no-install-recommends" ]; then
  extra=(--no-install-recommends)
  shift
fi

if [ "$#" -lt 1 ]; then
  echo "usage: apt-install.sh [--no-install-recommends] package [package...]" >&2
  exit 2
fi

if [ "$(id -u)" -eq 0 ]; then
  SUDO=()
else
  SUDO=(sudo)
fi

for attempt in 1 2 3 4 5; do
  if ! "${SUDO[@]}" apt-get update -qq; then
    echo "apt-get update failed (attempt ${attempt})" >&2
    sleep $((attempt * 8))
    continue
  fi
  set +e
  out=$("${SUDO[@]}" apt-get install -y -qq "${extra[@]}" "$@" 2>&1)
  status=$?
  set -e
  printf '%s\n' "$out"
  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  if printf '%s\n' "$out" | grep -q 'Unable to locate package\|has no installation candidate'; then
    exit 2
  fi
  echo "apt-get install failed (attempt ${attempt})" >&2
  sleep $((attempt * 8))
done

echo "apt-get failed after 5 attempts: $*" >&2
exit 1
