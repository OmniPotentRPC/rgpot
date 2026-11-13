#!/bin/sh
# Hand-written stand-in for the ORCA binary. Not an ORCA build and not
# the output of an ORCA run. rgpot's tests put this on PATH as `orca`.
# RGPOT_ORCA_FAKE_ENGRAD is the .engrad fixture to copy into place.
set -u

log=${RGPOT_ORCA_FAKE_LOG:-}
inp=
for a in "$@"; do
  case "$a" in
    *.inp) inp=$a ;;
  esac
done
if [ -z "$inp" ]; then
  echo "fake orca: no .inp argument" >&2
  exit 2
fi
base=${inp%.inp}
if [ -n "$log" ]; then
  printf '=== %s ===\n' "$inp" >> "$log"
  printf 'CWD %s\n' "$(pwd)" >> "$log"
  /bin/cat "$inp" >> "$log"
fi

if [ "${RGPOT_ORCA_FAKE_MOREAD_FAIL:-}" = 1 ] && /bin/grep -q 'MORead' "$inp"; then
  echo "ORCA finished by error termination in SCF"
  echo "SCF NOT CONVERGED AFTER 1 CYCLES"
  exit 1
fi
if [ -n "${RGPOT_ORCA_FAKE_SCF_FAIL:-}" ]; then
  echo "SCF NOT CONVERGED AFTER 5 CYCLES"
  exit 1
fi
if [ -n "${RGPOT_ORCA_FAKE_FAIL:-}" ]; then
  echo "ORCA finished by error termination"
  echo "fake orca forced failure"
  exit "${RGPOT_ORCA_FAKE_FAIL}"
fi
if [ -n "${RGPOT_ORCA_FAKE_BAD_ENGRAD:-}" ]; then
  echo "this is not an engrad file" > "${base}.engrad"
  exit 0
fi
replay=${RGPOT_ORCA_FAKE_REPLAY:-${ORCA_FAKE_REPLAY:-}}
if [ -n "$replay" ]; then
  /bin/cp "$replay/replay.engrad" "${base}.engrad"
  if [ -f "$replay/replay.stdout" ]; then
    /bin/cat "$replay/replay.stdout"
  fi
elif [ -z "${RGPOT_ORCA_FAKE_NO_ENGRAD:-}" ]; then
  src=${RGPOT_ORCA_FAKE_ENGRAD:-}
  if [ -z "$src" ]; then
    echo "fake orca: RGPOT_ORCA_FAKE_ENGRAD is unset" >&2
    exit 2
  fi
  /bin/cp "$src" "${base}.engrad"
fi
printf 'gbw' > "${base}.gbw"
exit 0
