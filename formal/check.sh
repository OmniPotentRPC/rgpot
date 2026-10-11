#!/bin/sh
# Fail on sorry, admit, an axiom declaration, or an axiom outside the
# standard kernel trio.
set -eu
cd "$(dirname "$0")"
lake build
if grep -RInE --include='*.lean' -e '\bsorry\b' -e '\badmit\b' Rpot ./*.lean; then
  echo "sorry or admit" >&2
  exit 1
fi
if grep -RInE --include='*.lean' -e '^[[:space:]]*axiom[[:space:]]' Rpot ./*.lean; then
  echo "axiom declaration" >&2
  exit 1
fi
out=$(lake env lean PrintAxioms.lean)
printf '%s\n' "$out"
printf '%s\n' "$out" | grep -q 'Rpot.twoSum_telescopes'
# The printed line is: 'name' depends on axioms: [a, b]
axioms=$(printf '%s\n' "$out" | sed -n 's/.*\[\(.*\)\].*/\1/p' | tr ',' '\n')
if [ -z "$axioms" ]; then
  echo "no axiom list" >&2
  exit 1
fi
for a in $axioms; do
  a=$(printf '%s' "$a" | tr -d ' ')
  case "$a" in
    propext|Classical.choice|Quot.sound) ;;
    *)
      echo "unexpected axiom: $a" >&2
      exit 1
      ;;
  esac
done
echo "formal check ok"
