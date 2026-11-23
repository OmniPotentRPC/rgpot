#!/usr/bin/env bash
# Build a double-precision, thread-MPI GROMACS with the installed nblib and
# legacy APIs as shared libraries, and install it under PREFIX. The result is
# what -Dwith_nblib=enabled -Dnblib_prefix=PREFIX needs.
#
# Usage: build_gromacs.sh PREFIX [VERSION]
#   VERSION defaults to 2025.5 and must be one of the pinned tarballs.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 PREFIX [VERSION]" >&2
  exit 2
fi
prefix=$1
version=${2:-2025.5}

case $version in
  2024.6) sha256=7cbad81f51c71a144d646515a7249aa74940b3f68071f51410e3a9473f05b339 ;;
  2025.5) sha256=7d0c7791675d0171af71dbfb35bfbb90bc4deb08266e62fbdcdac996f1f60174 ;;
  2026.4) sha256=58eda60979b124fcf9dff8b1b63cedfbc72fdb36d99e921dc1bdc310571098b6 ;;
  *) echo "no pinned tarball for GROMACS $version" >&2; exit 2 ;;
esac

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
tarball="$work/gromacs-$version.tar.gz"
curl -fsSL --retry 3 -o "$tarball" "https://ftp.gromacs.org/gromacs/gromacs-$version.tar.gz"
echo "$sha256  $tarball" | sha256sum -c -
tar xzf "$tarball" -C "$work"

cmake -S "$work/gromacs-$version" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_SHARED_LIBS=ON \
  -DGMX_INSTALL_NBLIB_API=ON \
  -DGMX_INSTALL_LEGACY_API=ON \
  -DGMX_DOUBLE=ON \
  -DGMX_MPI=OFF \
  -DGMX_GPU=OFF \
  -DGMX_FFT_LIBRARY=fftpack \
  -DGMX_EXTERNAL_BLAS=OFF \
  -DGMX_EXTERNAL_LAPACK=OFF \
  -DGMX_USE_PLUMED=OFF \
  -DGMX_NNPOT=OFF \
  -DGMX_HWLOC=OFF \
  -DGMX_BUILD_HELP=OFF \
  -DGMX_DEFAULT_SUFFIX=ON \
  -DBUILD_TESTING=OFF
cmake --build "$work/build" --target install -j "$(nproc 2>/dev/null || echo 2)"
"$prefix/bin/gmx_d" --version | grep -E 'GROMACS version|Precision'
test -n "$(find "$prefix" -name 'libnblib_gmx*' -print -quit)" ||
  { echo "libnblib_gmx was not installed under $prefix" >&2; exit 1; }
