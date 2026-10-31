#!/usr/bin/env bash
# Generate the libxckernel C package on a host listed in RGPOT_REGEN_HOSTS
# (comma-separated) only, unless XCKERNEL_ALLOW_REGEN=1.
# PIN gets "host ${RGPOT_HOST_LABEL}", or "host unspecified" when it is unset
# or blank.
# Families: lda,gga,mgga_tau,mgga_lapl ; max_order 2.
# Pin: d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4
set -euo pipefail

host=$(hostname -s 2>/dev/null || hostname)

_require_regen_host() {
  local h full
  local -a hosts
  full=$(hostname 2>/dev/null || echo "${host}")
  IFS=',' read -r -a hosts <<< "${RGPOT_REGEN_HOSTS:-}"
  for h in "${hosts[@]}"; do
    h="${h//[[:space:]]/}"
    if [[ -n "${h}" && ( "${h}" == "${host}" || "${h}" == "${full}" ) ]]; then
      return 0
    fi
  done
  if [[ "${XCKERNEL_ALLOW_REGEN:-}" == "1" ]]; then
    return 0
  fi
  echo "scripts/gen_libxckernel.sh: generate only on a host listed in" \
    "RGPOT_REGEN_HOSTS (got ${host}; XCKERNEL_ALLOW_REGEN=1 overrides)" >&2
  exit 2
}
_require_regen_host

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
python3 -m xckernel.catalog "${dest}" "lda,gga,mgga_tau,mgga_lapl" 2 c

# PIN records a neutral label, never the system hostname.
host_label=""
read -r host_label <<< "${RGPOT_HOST_LABEL:-}" || true
cat > "${dest}/PIN" <<EOF
libxckernel ${pin}
families lda,gga,mgga_tau,mgga_lapl
max_order 2
command python3 -m xckernel.catalog third_party/libxckernel lda,gga,mgga_tau,mgga_lapl 2 c
host ${host_label:-unspecified}
EOF

# Reject later-slice objects in the compiled TUs (manifest may mention GIAO skips).
if rg --files "${dest}/src" | rg '_o[34]|_giao|cmgga|hmgga'; then
  echo "vendor tree contains later-slice kernels" >&2
  exit 3
fi

echo "generated ${dest}"
find "${dest}/src" -name '*.cpp' | wc -l
