#!/bin/sh
# Launch RefusedSplit on 12 ranks. Open MPI needs --oversubscribe when
# the host has fewer slots than 12.
set -eu
if [ "$#" -ne 1 ]; then
  echo "usage: refused_split.sh EXE" >&2
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
# Open MPI 5 --help is a short usage line. --oversubscribe is documented
# in the man page and is what a 12-rank run needs on a smaller allocation.
if printf '%s\n' "$help" "$ver" | grep -q 'Open MPI'; then
  OVER=--oversubscribe
elif printf '%s\n' "$help" | grep -q oversubscribe; then
  OVER=--oversubscribe
fi
# shellcheck disable=SC2086
exec "$LAUNCH" -n 12 $OVER "$EXE"
