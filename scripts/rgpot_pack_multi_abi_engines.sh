#!/usr/bin/env bash
# Build libmetatomic_engine.so for each torch major and inject into a wheel as
#   rgpot/lib/torch-X.Y/libmetatomic_engine.so
#
# Product support floor: torch 2.7+. Defaults cover 2.7–2.13.
#
# Usage:
#   RGPOT_TORCH_MAJORS="2.7 2.9 2.13" ./scripts/rgpot_pack_multi_abi_engines.sh dist/rgpot-*.whl
#
# Requires: existing wheel, compilers, patchelf, pip network (or pre-seeded
# torch CPU wheels). Builds each ABI in an isolated prefix under $RGPOT_ABI_ROOT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHL="${1:?path to rgpot wheel}"
MAJORS="${RGPOT_TORCH_MAJORS:-2.7 2.8 2.9 2.10 2.11 2.12 2.13}"
ABI_ROOT="${RGPOT_ABI_ROOT:-$HOME/tmp/rgpot-multi-abi}"
PY="${RGPOT_BUILD_PYTHON:-python3}"
mkdir -p "$ABI_ROOT"
WORK=$(mktemp -d -p "${TMPDIR:-/tmp}")
trap 'rm -rf "$WORK"' EXIT
"$PY" -m zipfile -e "$WHL" "$WORK"

# Host purelib for metatomic multi-ABI headers/libs (same package for all majors)
HOST_PURE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [[ ! -d "$HOST_PURE/metatomic/torch" ]]; then
  echo "ERROR: metatomic-torch not installed in $PY purelib=$HOST_PURE" >&2
  exit 1
fi

build_one() {
  local maj="$1"
  local prefix="$ABI_ROOT/torch-$maj"
  local engdir="$WORK/rgpot/lib/torch-$maj"
  mkdir -p "$engdir" "$prefix"
  echo "=== build engine torch-$maj ==="
  # Install matching torch CPU into isolated prefix if missing
  if [[ ! -f "$prefix/torch/lib/libtorch_cpu.so" ]]; then
    echo "pip download/install torch $maj into $prefix"
    "$PY" -m pip install -q "torch==${maj}.*" \
      --index-url https://download.pytorch.org/whl/cpu \
      --target "$prefix" 2>&1 | tail -5 || \
    "$PY" -m pip install -q "torch==${maj}.*" --target "$prefix" 2>&1 | tail -5
  fi
  if [[ ! -f "$prefix/torch/lib/libtorch_cpu.so" ]]; then
    echo "SKIP torch-$maj: no libtorch_cpu.so" >&2
    return 0
  fi
  # Configure cmake prefix for this torch
  export CMAKE_PREFIX_PATH="$prefix/torch/share/cmake${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export PYTHONPATH="$prefix${PYTHONPATH:+:$PYTHONPATH}"
  # Build only metatomic engine via meson in a temp builddir
  local bdir="$ABI_ROOT/build-$maj"
  rm -rf "$bdir"
  # Point purelib probe at HOST for metatomic multi-abi + this torch for libtorch
  # meson uses python3 from PATH — use a small wrapper env
  (
    cd "$ROOT"
    # Prefer no-isolation style: meson setup with options
    export PATH="$(dirname "$PY"):$PATH"
    # Fake purelib for probe: use host so metatomic is found; torch lib from prefix
    # Root meson probes torch version from purelib/torch/version.py — need our maj
    mkdir -p "$prefix/torch"
    # Ensure version.py matches maj
    if [[ ! -f "$prefix/torch/version.py" ]]; then
      echo "__version__ = '${maj}.0'" > "$prefix/torch/version.py"
    fi
    # Symlink metatomic stack into prefix pure-style layout for includes
    for pkg in metatensor metatensor_torch metatomic vesin; do
      if [[ -d "$HOST_PURE/$pkg" && ! -e "$prefix/$pkg" ]]; then
        ln -sfn "$HOST_PURE/$pkg" "$prefix/$pkg"
      fi
    done
    export PYTHONPATH="$prefix:$HOST_PURE${PYTHONPATH:+:$PYTHONPATH}"
    "$PY" -c "import sys; print('py', sys.executable); import torch; print('torch', torch.__version__)" || true
    # Use meson directly for engine target only if full project configures
    meson setup "$bdir" \
      -Dwith_python=false \
      -Dwith_rpc=false \
      -Dwith_tests=false \
      -Dwith_examples=false \
      -Dwith_metatomic=true \
      -Dwith_xtb=false \
      -Dwith_tblite=false \
      -Dwith_rust_core=false \
      -Dwith_cache=false \
      -Dwith_eigen=false \
      -Db_lto=false \
      --prefix="$prefix/install" 2>&1 | tail -40
    meson compile -C "$bdir" metatomic_engine 2>&1 | tail -30
    SO=$(find "$bdir" -name 'libmetatomic_engine.so' | head -1)
    if [[ -z "$SO" ]]; then
      echo "FAIL no engine for $maj" >&2
      return 1
    fi
    cp -a "$SO" "$engdir/librgpot_metatomic_engine.so"
    ln -sfn librgpot_metatomic_engine.so "$engdir/libmetatomic_engine.so"
    # RPATH for multi-ABI layout under rgpot/lib/torch-X.Y/
    peer='$ORIGIN/../../../'
    rpath="\$ORIGIN:${peer}torch/lib:${peer}metatensor/lib:${peer}vesin/lib"
    rpath+=":${peer}metatensor_torch/torch-${maj}/lib"
    rpath+=":${peer}metatomic/torch/torch-${maj}/lib"
    rpath+=":${peer}metatensor/torch/torch-${maj}/lib"
    patchelf --set-rpath "$rpath" "$engdir/librgpot_metatomic_engine.so"
    echo "OK torch-$maj -> $engdir/librgpot_metatomic_engine.so"
  )
}

for maj in $MAJORS; do
  build_one "$maj" || echo "WARN skip $maj"
done

# Also copy primary mesonpy engine into matching torch dir if present
PRIMARY=$(find "$WORK" -path '*mesonpy.libs/libmetatomic_engine.so' | head -1 || true)
if [[ -n "$PRIMARY" ]]; then
  pmaj="${RGPOT_TORCH_MAJOR:-}"
  if [[ -z "$pmaj" ]]; then
    pmaj=$(readelf -d "$PRIMARY" | sed -n 's/.*torch-\([0-9]\+\.[0-9]\+\).*/\1/p' | head -1 || true)
  fi
  pmaj=${pmaj:-2.9}
  mkdir -p "$WORK/rgpot/lib/torch-$pmaj"
  if [[ ! -f "$WORK/rgpot/lib/torch-$pmaj/librgpot_metatomic_engine.so" ]]; then
    cp -a "$PRIMARY" "$WORK/rgpot/lib/torch-$pmaj/librgpot_metatomic_engine.so"
    ln -sfn librgpot_metatomic_engine.so \
      "$WORK/rgpot/lib/torch-$pmaj/libmetatomic_engine.so"
    peer='$ORIGIN/../../../'
    rpath="\$ORIGIN:${peer}torch/lib:${peer}metatensor/lib:${peer}vesin/lib"
    rpath+=":${peer}metatensor_torch/torch-${pmaj}/lib"
    rpath+=":${peer}metatomic/torch/torch-${pmaj}/lib"
    rpath+=":${peer}metatensor/torch/torch-${pmaj}/lib"
    patchelf --set-rpath "$rpath" "$WORK/rgpot/lib/torch-$pmaj/librgpot_metatomic_engine.so"
    echo "copied primary engine to torch-$pmaj"
  fi
fi

echo "bundled engines:"
find "$WORK/rgpot/lib" -name 'libmetatomic_engine.so' 2>/dev/null || true

python3 - <<PY
import zipfile
from pathlib import Path
root = Path("$WORK")
out = Path("$WHL")
tmp = out.with_suffix(".whl.tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root).as_posix())
tmp.replace(out)
print("MULTI_ABI_PACK_OK", out)
PY
# final repair pass
bash "$ROOT/scripts/rgpot_repair_wheel.sh" "$WHL"
