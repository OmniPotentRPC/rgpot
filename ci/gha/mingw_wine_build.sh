#!/usr/bin/env bash
# MIT License
# Copyright 2023--present rgpot developers
# MinGW-w64 cross build of rgpot with the tests run under wine.
# Usage: ci/gha/mingw_wine_build.sh BUILD_DIR [meson setup options...]
# Needs the Debian/Ubuntu mingw-w64 posix compilers (C, C++, Fortran),
# wine, cmake, ninja, meson, and rustup's x86_64-pc-windows-gnu target.
# Catch2 is built for the target into BUILD_DIR-deps on the first run.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build=$(realpath -m "$1")
shift
deps="${build}-deps"
triple=x86_64-w64-mingw32
catch2_version=v3.7.1

winpath() {
  printf 'Z:%s' "${1//\//\\}"
}
# libgfortran, libgomp, libstdc++ and winpthread DLLs for every test and
# for meson's sanity check of the Fortran compiler.
gcclib=$(dirname "$("$triple-gcc-posix" -print-libgcc-file-name)")
WINEPATH="$(winpath "$gcclib");$(winpath "/usr/$triple/lib")"
WINEDEBUG=-all
# No wine-mono or wine-gecko install prompt on a fresh prefix.
WINEDLLOVERRIDES='mscoree,mshtml='
export WINEPATH WINEDEBUG WINEDLLOVERRIDES

if [ ! -f "$deps/lib/cmake/Catch2/Catch2Config.cmake" ]; then
  rm -rf "$deps"
  mkdir -p "$deps"
  git clone --quiet --depth 1 --branch "$catch2_version" \
    https://github.com/catchorg/Catch2.git "$deps/src"
  cat > "$deps/toolchain.cmake" <<EOF
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER $triple-gcc-posix)
set(CMAKE_CXX_COMPILER $triple-g++-posix)
set(CMAKE_RC_COMPILER $triple-windres)
set(CMAKE_FIND_ROOT_PATH /usr/$triple)
EOF
  cmake -S "$deps/src" -B "$deps/build" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$deps/toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$deps" \
    -DBUILD_TESTING=OFF \
    -DCATCH_INSTALL_DOCS=OFF \
    -DCATCH_INSTALL_EXTRAS=OFF
  cmake --build "$deps/build" --target install
fi

meson setup "$build" "$root" \
  --cross-file "$root/ci/gha/machines/mingw64-wine.ini" \
  -Dcmake_prefix_path="$deps" \
  "$@"
meson compile -C "$build"

# AotiPublicationStressTest holds model.pt2 open and replaces it.
# Wine's unix drive returns access denied for that replace: it has no
# POSIX rename. NTFS accepts it, and the native Windows job still runs
# the test.
wine_only=(
  AotiPublicationStressTest
)
mapfile -t wine_tests < <(meson test -C "$build" --list --no-rebuild)
run=()
for name in "${wine_tests[@]}"; do
  short="${name##*:}"
  skip=0
  for held in "${wine_only[@]}"; do
    if [[ "$short" == "$held" ]]; then
      skip=1
      break
    fi
  done
  if [[ "$skip" -eq 0 ]]; then
    run+=("$short")
  fi
done
echo "==> skipping under wine: ${wine_only[*]}"
if [[ ${#run[@]} -eq 0 ]]; then
  echo "wine test list is empty" >&2
  exit 1
fi
for held in "${wine_only[@]}"; do
  for name in "${run[@]}"; do
    if [[ "$name" == "$held" ]]; then
      echo "wine skip list did not remove ${held}" >&2
      exit 1
    fi
  done
done
meson test -C "$build" --print-errorlogs "${run[@]}"
