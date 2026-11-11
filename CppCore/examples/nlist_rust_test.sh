#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
# Compile the hand-written Rust caller against the neighbour archives.
set -eu
root=$1
build=$2
rustc_bin=rustc
if ! "$rustc_bin" --version >/dev/null 2>&1; then
  rustc_bin=$(find /usr/local/rustup/toolchains "${RUSTUP_HOME:-/usr/local/rustup}/toolchains" \
    -path '*/bin/rustc' -type f -print -quit 2>/dev/null || true)
fi
if [ -z "$rustc_bin" ]; then
  echo "rustc is not available" >&2
  exit 1
fi
half=$(find "$build" -name 'librgpot_halfshell.a' -print -quit)
mini=$(find "$build" -name 'librgpot_minimage.a' -print -quit)
fma=$(find "$build" -name 'librgpot_exact_fma.a' -print -quit)
dir=$(mktemp -d)
g++ -shared -o "$dir/libnlist.so" -Wl,--whole-archive "$half" "$mini" "$fma" \
  -Wl,--no-whole-archive -fopenmp
out=$(mktemp)
"$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
  -L "$dir" -lnlist
LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out"
rm -rf "$dir" "$out"
