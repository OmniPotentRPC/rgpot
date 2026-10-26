#!/bin/sh
# Launch CollectiveCacheTest on four ranks under a 30 s bound: a hang in
# the potential's collective shows up as the timeout's status 124.
set -eu
if [ "$#" -ne 1 ]; then
  echo "usage: collective_cache.sh EXE" >&2
  exit 2
fi
EXE=$1
if command -v mpirun >/dev/null 2>&1; then
  LAUNCH=mpirun
elif command -v mpiexec >/dev/null 2>&1; then
  LAUNCH=mpiexec
else
  echo "mpirun or mpiexec is required" >&2
  exit 1
fi
help=$("$LAUNCH" --help 2>&1 || true)
ver=$("$LAUNCH" --version 2>&1 || true)
OVER=
if printf '%s\n' "$help" "$ver" | grep -q 'Open MPI'; then
  OVER=--oversubscribe
elif printf '%s\n' "$help" | grep -q oversubscribe; then
  OVER=--oversubscribe
fi
if command -v timeout >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  exec timeout 30 "$LAUNCH" -n 4 $OVER "$EXE"
fi
# shellcheck disable=SC2086
exec "$LAUNCH" -n 4 $OVER "$EXE"
