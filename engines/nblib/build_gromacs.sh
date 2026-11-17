#!/usr/bin/env bash
# Build the pinned GROMACS release with the nblib API and install it under
# PREFIX: a double-precision, serial build with shared libraries, the
# installed nblib headers and the legacy headers they include. rgpot's
# -Dwith_nblib=true -Dnblib_prefix=PREFIX links librgpot_nblib against it,
# and PREFIX/bin/gmx_d is the reference the mdrun -rerun tests compare with.
#
# Usage: build_gromacs.sh PREFIX [VERSION]
#   VERSION is 2025.5 (the default), whose nblib API this library targets.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 PREFIX [VERSION]" >&2
  exit 2
fi
prefix=$1
version=${2:-2025.5}

case $version in
  2025.5) sha256=7d0c7791675d0171af71dbfb35bfbb90bc4deb08266e62fbdcdac996f1f60174 ;;
  *) echo "no pinned tarball for GROMACS $version" >&2; exit 2 ;;
esac

work=$(mktemp -d)
echo "building in $work"
tarball="$work/gromacs-$version.tar.gz"
curl -fsSL --retry 3 -o "$tarball" "https://ftp.gromacs.org/gromacs/gromacs-$version.tar.gz"
echo "$sha256  $tarball" | sha256sum -c -
tar xzf "$tarball" -C "$work"

cmake -S "$work/gromacs-$version" -B "$work/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$prefix" \
  -DBUILD_SHARED_LIBS=ON \
  -DGMX_DOUBLE=ON \
  -DGMX_MPI=OFF \
  -DGMX_OPENMP=OFF \
  -DGMX_GPU=OFF \
  -DGMX_SIMD=None \
  -DGMX_FFT_LIBRARY=fftpack \
  -DGMX_EXTERNAL_BLAS=OFF \
  -DGMX_EXTERNAL_LAPACK=OFF \
  -DGMX_HWLOC=OFF \
  -DGMX_BUILD_HELP=OFF \
  -DGMX_USE_PLUMED=OFF \
  -DGMX_NNPOT=OFF \
  -DGMX_INSTALL_NBLIB_API=ON \
  -DGMX_INSTALL_LEGACY_API=ON \
  -DBUILD_TESTING=OFF
cmake --build "$work/build" --target install -j "$(nproc 2>/dev/null || echo 2)"
"$prefix/bin/gmx_d" --version | grep -E 'GROMACS version|Precision'
test -f "$prefix/include/nblib/simulationstate.h"
ls "$prefix"/lib*/libnblib_gmx.so
