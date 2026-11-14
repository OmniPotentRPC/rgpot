#!/bin/sh
# MIT License
# Copyright 2023--present rgpot developers
# Compile the hand-written Rust caller against the neighbour archives.
# Usage: nlist_rust_test.sh ROOT BUILD [WRAPPER...]
# RGPOT_RUST_TARGET set to a windows-gnu triple builds a MinGW DLL with
# $CXX, links the Rust caller with $CXX as well, and runs it under
# WRAPPER (a cross build's exe_wrapper).
set -eu
root=$1
build=$2
shift 2
target=${RGPOT_RUST_TARGET:-}
rustc_bin=rustc
if ! "$rustc_bin" --version >/dev/null 2>&1; then
  rustc_bin=$(find /usr/local/rustup/toolchains "${RUSTUP_HOME:-/usr/local/rustup}/toolchains" \
    -path '*/bin/rustc' -type f -print -quit 2>/dev/null || true)
fi
if [ -z "$rustc_bin" ]; then
  echo "rustc is not available" >&2
  exit 1
fi
find_archive() {
  find "$build" \( -name "$1" -o -name "$2" -o -name "$3" \) -print -quit
}
half=$(find_archive librgpot_halfshell.a librgpot_halfshell.lib rgpot_halfshell.lib)
mini=$(find_archive librgpot_minimage.a librgpot_minimage.lib rgpot_minimage.lib)
fma=$(find_archive librgpot_exact_fma.a librgpot_exact_fma.lib rgpot_exact_fma.lib)
if [ -z "$half" ] || [ -z "$mini" ] || [ -z "$fma" ]; then
  echo "neighbour archives were not built" >&2
  exit 1
fi
cxx=${CXX:-c++}
os=$(uname -s)
case "$target" in
  *-windows-gnu*) os=windows-gnu ;;
esac
omp_flag=
if "$cxx" -fopenmp -x c++ -E - >/dev/null 2>&1 <<'EOF'
int main() { return 0; }
EOF
then
  omp_flag=-fopenmp
fi
# Coverage builds instrument the archives. The shared library has to pull
# the gcov runtime or the loader cannot resolve __gcov_init.
gcov_flag=
if nm "$half" 2>/dev/null | grep -q __gcov_init; then
  gcov_flag=--coverage
fi
dir=$(mktemp -d)
case "$os" in
  windows-gnu)
    # -static keeps libgomp, libstdc++ and winpthread inside nlist.dll,
    # so the loader needs only the DLL beside the executable.
    "$cxx" -shared -static -o "$dir/nlist.dll" -Wl,--out-implib,"$dir/libnlist.dll.a" \
      -Wl,--whole-archive "$half" "$mini" "$fma" -Wl,--no-whole-archive \
      $omp_flag $gcov_flag
    if command -v cygpath >/dev/null 2>&1; then
      wdir=$(cygpath -w "$dir")
      TMP="$wdir"
      TEMP="$wdir"
      TMPDIR="$wdir"
      export TMP TEMP TMPDIR
    fi
    ;;
  Darwin)
    "$cxx" -shared -o "$dir/libnlist.dylib" \
      -Wl,-force_load,"$half" -Wl,-force_load,"$mini" -Wl,-force_load,"$fma" \
      $omp_flag $gcov_flag
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # cl rejects -Wl,--whole-archive. link.exe pulls every object; the
    # import library is nlist.lib, which rustc's -lnlist finds.
    # Path conversion stays on for rustc. Only the link.exe line opts out,
    # or Git bash rewrites /WHOLEARCHIVE. rustc is a Windows binary and
    # creates its metadata dir from TMP; an MSYS /tmp does not exist there.
    whalf=$(cygpath -w "$half")
    wmini=$(cygpath -w "$mini")
    wfma=$(cygpath -w "$fma")
    wdir=$(cygpath -w "$dir")
    MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
      link.exe /nologo /DLL "/OUT:${wdir}\\nlist.dll" \
      "/WHOLEARCHIVE:${whalf}" "/WHOLEARCHIVE:${wmini}" "/WHOLEARCHIVE:${wfma}" \
      /EXPORT:rgpot_nlist_build \
      /EXPORT:rgpot_nlist_request_init \
      /EXPORT:rgpot_nlist_strerror
    if [ ! -f "$dir/nlist.lib" ]; then
      echo "link.exe did not write nlist.lib" >&2
      ls -la "$dir" >&2 || true
      exit 1
    fi
    TMP="$wdir"
    TEMP="$wdir"
    TMPDIR="$wdir"
    export TMP TEMP TMPDIR
    ;;
  *)
    "$cxx" -shared -o "$dir/libnlist.so" -Wl,--whole-archive "$half" "$mini" "$fma" \
      -Wl,--no-whole-archive $omp_flag $gcov_flag
    ;;
esac
out=$(mktemp)
case "$os" in
  windows-gnu)
    rm -f "$out"
    out="$dir/nlist_rust.exe"
    # Under MSYS2, rustc is a Windows binary and takes Windows paths.
    rmain="$root/CppCore/examples/nlist_rust/main.rs"
    rdir=$dir
    rout=$out
    if command -v cygpath >/dev/null 2>&1; then
      rmain=$(cygpath -m "$rmain")
      rdir=$(cygpath -m "$dir")
      rout=$(cygpath -m "$out")
    fi
    "$rustc_bin" --edition 2021 --target "$target" -C "linker=$cxx" \
      "$rmain" -o "$rout" -L "native=$rdir" -lnlist
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # rustc's MSVC linker looks for nlist.lib on a native search path.
    # An MSYS -L directory is not one it opens.
    wout=$(cygpath -w "$out")
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$wout" \
      -L "native=${wdir}" -lnlist
    ;;
  *)
    "$rustc_bin" --edition 2021 "$root/CppCore/examples/nlist_rust/main.rs" -o "$out" \
      -L "$dir" -lnlist
    ;;
esac
case "$os" in
  windows-gnu)
    "$@" "$out"
    ;;
  Darwin)
    DYLD_LIBRARY_PATH="$dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" "$out"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    PATH="$dir:$PATH" "$out"
    ;;
  *)
    LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$out"
    ;;
esac
rm -rf "$dir" "$out"
