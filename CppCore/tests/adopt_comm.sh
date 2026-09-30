#!/bin/sh
# Launch AdoptComm. The profile library is assigned on the ranks so the
# launcher does not load MPI_Comm_split itself.
set -eu
if [ "$#" -ne 4 ]; then
  echo "usage: adopt_comm.sh MODE EXE ENGINE PROFILE" >&2
  exit 2
fi
MODE=$1
EXE=$2
ENGINE=$3
PROFILE=$4
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
case "$MODE" in
  missing) N=2 ;;
  split) N=4 ;;
  *)
    echo "mode must be missing or split" >&2
    exit 2
    ;;
esac
FWD=
if printf '%s\n' "$help" "$ver" | grep -q 'Open MPI'; then
  OVER=--oversubscribe
  FWD="-x LD_PRELOAD=$PROFILE"
elif printf '%s\n' "$help" "$ver" | grep -Eq 'MPICH|Hydra'; then
  FWD="-env LD_PRELOAD $PROFILE"
elif printf '%s\n' "$help" | grep -q oversubscribe; then
  OVER=--oversubscribe
  FWD="-x LD_PRELOAD=$PROFILE"
fi
# shellcheck disable=SC2086
exec "$LAUNCH" -n "$N" $OVER $FWD "$EXE" "$MODE" "$ENGINE"
