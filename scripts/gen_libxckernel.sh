#!/usr/bin/env bash
# Generate the first-slice libxckernel C package on rg.terra only.
# Families: lda,gga,mgga_tau ; max_order 2.
# Pin: d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4
set -euo pipefail

host=$(hostname -s 2>/dev/null || hostname)
if [[ "${host}" != "rgam5terra" && "${XCKERNEL_ALLOW_REGEN:-}" != "1" ]]; then
  echo "scripts/gen_libxckernel.sh: generate only on rg.terra (got ${host})" >&2
  exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd)
dest="${root}/third_party/libxckernel"
pin="d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4"
src="${LIBXCKERNEL_SRC:-}"
if [[ -z "${src}" ]]; then
  if [[ -d /tmp/libxckernel-src/xckernel ]]; then
    src=/tmp/libxckernel-src
  elif [[ -d "${root}/subprojects/libxckernel-d6a9d57/xckernel" ]]; then
    src="${root}/subprojects/libxckernel-d6a9d57"
  else
    echo "set LIBXCKERNEL_SRC to the pinned libxckernel checkout" >&2
    exit 2
  fi
fi

export PYTHONPATH="${src}${PYTHONPATH:+:${PYTHONPATH}}"
python3 - <<'PY' || { echo "need sympy+numpy in the active env (pixi -e xckernel)" >&2; exit 2; }
import numpy, sympy
print("numpy", numpy.__version__, "sympy", sympy.__version__)
PY

if [[ -e "${dest}" ]]; then
  rtrash -rf "${dest}"
fi
mkdir -p "${dest}"
python3 -m xckernel.catalog "${dest}" "lda,gga,mgga_tau" 2 c

cat > "${dest}/PIN" <<EOF
libxckernel ${pin}
families lda,gga,mgga_tau
max_order 2
command python3 -m xckernel.catalog third_party/libxckernel lda,gga,mgga_tau 2 c
host ${host}
EOF

# Reject later-slice objects in the compiled TUs (manifest may mention GIAO skips).
if ls "${dest}/src" | grep -E '_o[34]|_giao|cmgga|hmgga|mgga_lapl'; then
  echo "first-slice vendor tree contains later-slice kernels" >&2
  exit 3
fi

echo "generated ${dest}"
find "${dest}/src" -name '*.cpp' | wc -l
