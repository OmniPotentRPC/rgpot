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
cxx=${CXX:-c++}
omp_flag=
if "$cxx" -fopenmp -x c++ -E - >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
then
  omp_flag=-fopenmp
fi
dir=$(mktemp -d)
if [ "$(uname -s)" = Darwin ]; then
  "$cxx" -shared -o "$dir/libnlist.dylib" \
    -Wl,-force_load,"$half" -Wl,-force_load,"$mini" -Wl,-force_load,"$fma" \
    $omp_flag
else
  "$cxx" -shared -o "$dir/libnlist.so" -Wl,--whole-archive "$half" "$mini" "$fma" \
    -Wl,--no-whole-archive $omp_flag
fi
out=$(mktemp)
"$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
  -L "$dir" -lnlist
if [ "$(uname -s)" = Darwin ]; then
  DYLD_LIBRARY_PATH="$dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" "$out"
else
  LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out"
fi
rm -rf "$dir" "$out"
