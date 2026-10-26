#!/bin/sh
# Launch the rank-exception binary under two ranks, or check who finalizes MPI.
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: rank_exception_abort.sh EXE MODE" >&2
  exit 2
fi

exe=$1
mode=$2

if command -v mpirun >/dev/null 2>&1; then
  launcher=mpirun
elif command -v mpiexec >/dev/null 2>&1; then
  launcher=mpiexec
else
  echo "mpirun or mpiexec is required" >&2
  exit 1
fi

run_bounded() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 30 "$@"
  else
    "$@"
  fi
}

help=$("$launcher" --help 2>&1 || true)
ver=$("$launcher" --version 2>&1 || true)
# Open MPI 5 --help does not list --oversubscribe. The branches below
# read that word from $help.
if printf '%s\n' "$help" "$ver" | grep -q 'Open MPI'; then
  help="$help
--oversubscribe"
fi

if [ "$mode" = "abort" ]; then
  out=$(mktemp -d)
  trap 'rm -rf "$out"' EXIT
  export RGPOT_RANK_LOG_DIR="$out"
  rc=0
  set +e
  if printf '%s\n' "$help" | grep -q -- '--oversubscribe'; then
    run_bounded "$launcher" -n 2 --oversubscribe \
      -x RGPOT_RANK_LOG_DIR -x RGPOT_CPMD_ENGINE \
      "$exe" abort
  else
    run_bounded "$launcher" -n 2 "$exe" abort
  fi
  rc=$?
  set -e
  log="$out/rank-1.log"
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]; then
    echo "abort job status $rc" >&2
    exit 1
  fi
  if [ ! -f "$log" ]; then
    echo "missing rank 1 log $log" >&2
    ls -la "$out" >&2
    exit 1
  fi
  if ! grep -q 'nAtoms must be positive' "$log"; then
    echo "rank 1 log missing rank 0 error" >&2
    cat "$log" >&2
    exit 1
  fi
  exit 0
fi

if [ "$mode" = "uneven" ]; then
  out=$(mktemp -d)
  trap 'rm -rf "$out"' EXIT
  export RGPOT_RANK_LOG_DIR="$out"
  rc=0
  set +e
  if printf '%s\n' "$help" | grep -q -- '--oversubscribe'; then
    run_bounded "$launcher" -n 4 --oversubscribe \
      -x RGPOT_RANK_LOG_DIR -x RGPOT_CPMD_ENGINE \
      "$exe" uneven
  else
    run_bounded "$launcher" -n 4 "$exe" uneven
  fi
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "uneven batch status $rc (124 is the 30 s hang bound)" >&2
    cat "$out"/rank-*.log >&2 || true
    exit 1
  fi
  for rank in 0 1 2 3; do
    if ! grep -q 'uneven ok' "$out/rank-$rank.log"; then
      echo "rank $rank did not finish the uneven batch" >&2
      cat "$out"/rank-*.log >&2 || true
      exit 1
    fi
  done
  exit 0
fi

if [ "$mode" = "share-bad" ] || [ "$mode" = "share-ok" ]; then
  out=$(mktemp -d)
  trap 'rm -rf "$out"' EXIT
  export RGPOT_RANK_LOG_DIR="$out"
  rc=0
  set +e
  if printf '%s\n' "$help" | grep -q -- '--oversubscribe'; then
    run_bounded "$launcher" -n 2 --oversubscribe \
      -x RGPOT_RANK_LOG_DIR -x RGPOT_CPMD_ENGINE \
      "$exe" "$mode"
  else
    run_bounded "$launcher" -n 2 "$exe" "$mode"
  fi
  rc=$?
  set -e
  if [ "$mode" = "share-ok" ]; then
    if [ "$rc" -ne 0 ]; then
      echo "share-ok status $rc" >&2
      cat "$out"/rank-*.log >&2 || true
      exit 1
    fi
    if ! grep -q 'share-ok 1.000000 -2.000000 3.500000 0.250000' "$out/rank-1.log"; then
      echo "rank 1 buffer does not match rank 0" >&2
      cat "$out"/rank-*.log >&2 || true
      exit 1
    fi
    exit 0
  fi
  if [ "$rc" -eq 0 ] || [ "$rc" -eq 124 ]; then
    echo "share-bad status $rc" >&2
    exit 1
  fi
  for ranklog in "$out/rank-0.log" "$out/rank-1.log"; do
    if [ ! -f "$ranklog" ] || ! grep -q 'shareFromCalculator owner out of range' "$ranklog"; then
      echo "missing shared error in $ranklog" >&2
      ls -la "$out" >&2
      cat "$out"/rank-*.log >&2 || true
      exit 1
    fi
  done
  exit 0
fi

if [ "$mode" = "finalize-owner" ]; then
  log=$(mktemp)
  trap 'rm -f "$log"' EXIT
  set +e
  if printf '%s\n' "$help" | grep -q -- '--oversubscribe'; then
    RGPOT_MPI_FINALIZE_TRACE=1 "$launcher" -n 1 \
      -x RGPOT_MPI_FINALIZE_TRACE "$exe" finalize-owner >"$log" 2>&1
  else
    RGPOT_MPI_FINALIZE_TRACE=1 "$launcher" -n 1 "$exe" finalize-owner >"$log" 2>&1
  fi
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "owner finalize status $rc" >&2
    cat "$log" >&2
    exit 1
  fi
  if ! grep -q 'rgpot MPI_Finalize' "$log"; then
    echo "owner finalize did not run" >&2
    cat "$log" >&2
    exit 1
  fi
  exit 0
fi

if [ "$mode" = "finalize-guest" ]; then
  log=$(mktemp)
  trap 'rm -f "$log"' EXIT
  set +e
  "$launcher" -n 1 "$exe" finalize-guest >"$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "guest finalize status $rc" >&2
    cat "$log" >&2
    exit 1
  fi
  if grep -q 'rgpot MPI_Finalize' "$log"; then
    echo "guest run finalized MPI" >&2
    exit 1
  fi
  exit 0
fi

echo "unknown mode $mode" >&2
exit 2
