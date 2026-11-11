#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
# Compile the hand-written Rust caller against the neighbour archives.
set -eu
root=$1
build=$2
half=$(find "$build" -name 'librgpot_halfshell.a' -print -quit)
mini=$(find "$build" -name 'librgpot_minimage.a' -print -quit)
fma=$(find "$build" -name 'librgpot_exact_fma.a' -print -quit)
out=$(mktemp)
rustc --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
  -C link-arg="$half" -C link-arg="$mini" -C link-arg="$fma" -l stdc++ -l m
"$out"
rm -f "$out"
