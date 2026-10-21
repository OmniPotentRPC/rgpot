#!/usr/bin/env bash
# Rewrite absolute purelib RUNPATHs to $ORIGIN-relative site-packages paths.
#
# Engines live at either:
#   .rgpot.mesonpy.libs/libmetatomic_engine.so          (legacy single)
#   rgpot/lib/torch-X.Y/libmetatomic_engine.so          (multi-ABI)
# Each engine gets RUNPATH for *its* torch-X.Y only (never list all ABIs).
set -euo pipefail
# --refresh-soname recopies librgpot.so.<version> onto the SONAME aliases
# after auditwheel. The pre-auditwheel copy lets ldd resolve librgpot.so.N,
# but auditwheel rewrites NEEDED only on the versioned file. _core loads the
# alias, which would still NEEDED libopenblas.so.0.
REFRESH=0
if [[ "${1:-}" == "--refresh-soname" ]]; then
  REFRESH=1
  shift
fi
WHL="${1:?wheel path}"
command -v readelf >/dev/null
if [[ "$REFRESH" -eq 0 ]]; then
  command -v patchelf >/dev/null
fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 -m zipfile -e "$WHL" "$WORK"

engine_rpath_for_maj() {
  local maj="$1"
  # From rgpot/lib/torch-X.Y/ → site-packages peers are ../../../
  # From .rgpot.mesonpy.libs/ → site-packages peers are ../
  local origin_to_site="$2"
  echo "\$ORIGIN:\${origin_to_site}torch/lib:\${origin_to_site}metatensor/lib:\${origin_to_site}vesin/lib:\${origin_to_site}metatensor_torch/torch-${maj}/lib:\${origin_to_site}metatomic/torch/torch-${maj}/lib:\${origin_to_site}metatensor/torch/torch-${maj}/lib"
}

CORE_RPATH='$ORIGIN:$ORIGIN/../.rgpot.mesonpy.libs:$ORIGIN/lib'
# librgpot lives under .rgpot.mesonpy.libs/. $ORIGIN alone cannot see
# site-packages/torch. RGPOT_TORCH_MAJOR adds the metatensor-torch ABI dir.
_maj="${RGPOT_TORCH_MAJOR:-}"
UMBRELLA_RPATH='$ORIGIN:$ORIGIN/../torch/lib:$ORIGIN/../metatensor/lib:$ORIGIN/../vesin/lib'
if [[ -n "$_maj" ]]; then
  UMBRELLA_RPATH+=":\$ORIGIN/../metatensor_torch/torch-${_maj}/lib"
  UMBRELLA_RPATH+=":\$ORIGIN/../metatomic/torch/torch-${_maj}/lib"
  UMBRELLA_RPATH+=":\$ORIGIN/../metatensor/torch/torch-${_maj}/lib"
fi

if [[ "$REFRESH" -eq 0 ]]; then
while IFS= read -r -d '' so; do
  base=$(basename "$so")
  rel=${so#"$WORK/"}
  if [[ "$base" == *metatomic_engine* ]]; then
    maj=""
    if [[ "$rel" =~ torch-([0-9]+\.[0-9]+)/ ]]; then
      maj="${BASH_REMATCH[1]}"
      # rgpot/lib/torch-X.Y/lib.so → three levels up to site-packages
      peer='$ORIGIN/../../../'
    else
      maj="${RGPOT_TORCH_MAJOR:-}"
      if [[ -z "$maj" ]]; then
        maj=$(readelf -d "$so" 2>/dev/null | sed -n 's/.*torch-\([0-9]\+\.[0-9]\+\).*/\1/p' | head -1 || true)
      fi
      maj=${maj:-2.9}
      peer='$ORIGIN/../'
    fi
    # Build rpath string without nested expansion bugs
    ENG_RPATH="\$ORIGIN:${peer}torch/lib:${peer}metatensor/lib:${peer}vesin/lib"
    ENG_RPATH+=":${peer}metatensor_torch/torch-${maj}/lib"
    ENG_RPATH+=":${peer}metatomic/torch/torch-${maj}/lib"
    ENG_RPATH+=":${peer}metatensor/torch/torch-${maj}/lib"
    echo "patchelf engine $rel (torch-$maj)"
    patchelf --set-rpath "$ENG_RPATH" "$so"
  elif [[ "$base" == _core* ]]; then
    echo "patchelf core $(basename "$so")"
    patchelf --set-rpath "$CORE_RPATH" "$so"
  elif [[ "$base" == librgpot.so* ]]; then
    echo "patchelf umbrella $rel"
    patchelf --set-rpath "$UMBRELLA_RPATH" "$so"
  elif [[ "$so" == *".rgpot.mesonpy.libs"* ]]; then
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
  fi
done < <(find "$WORK" -type f \( -name '*.so' -o -name '*.so.*' \) -print0)
fi

# zip/pip cannot store ELF soname symlinks. meson installs
# librgpot.so.<version> whose DT_SONAME is librgpot.so.<soversion>
# (kCxxSoversion). _core NEEDs that SONAME. Copy (not ln) so a second
# repair pass and pip extract both see real files.
LIBS=$(find "$WORK" -type d -name '.rgpot.mesonpy.libs' -print -quit)
if [[ -n "$LIBS" ]]; then
  real=$(find "$LIBS" -maxdepth 1 -type f -name 'librgpot.so.*.*' | head -1 || true)
  if [[ -n "$real" ]]; then
    soname=$(readelf -d "$real" | sed -n 's/.*SONAME.*\[\([^]]*\)\].*/\1/p' | head -1)
    [[ -n "$soname" ]] || { echo "no SONAME in $real" >&2; exit 1; }
    cp -f "$real" "$LIBS/$soname"
    cp -f "$real" "$LIBS/librgpot.so"
    echo "soname copies $(basename "$real") -> $soname + librgpot.so"
  fi
fi

bad=0
while IFS= read -r -d '' so; do
  rp=$(readelf -d "$so" 2>/dev/null | sed -n 's/.*Library \(runpath\|rpath\): \[\(.*\)\]/\2/p' || true)
  if echo "$rp" | grep -E '/home/|/Users/|/bbdir|/\.mesonpy' >/dev/null 2>&1; then
    echo "ERROR: absolute host path in $(basename "$so"): $rp" >&2
    bad=1
  fi
done < <(find "$WORK" -type f -name '*.so' -print0)
[[ $bad -eq 0 ]] || exit 1

python3 - <<PY
import base64, hashlib, zipfile
from pathlib import Path
root = Path("$WORK")
out = Path("$WHL")
records = list(root.glob("*.dist-info/RECORD"))
if records:
    rec = records[0]
    lines = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "RECORD":
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            lines.append(f"{rel},sha256={digest},{len(data)}")
    lines.append(f"{rec.relative_to(root).as_posix()},,")
    rec.write_text("\n".join(lines) + "\n")
tmp = out.with_suffix(".whl.tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root).as_posix())
tmp.replace(out)
print("REPAIR_OK", out)
PY
