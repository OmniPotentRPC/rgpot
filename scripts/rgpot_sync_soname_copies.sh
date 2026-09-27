#!/usr/bin/env bash
# Refresh the librgpot soname copies from the repaired library, after
# auditwheel repair.
#
# Wheels cannot store the ELF soname symlinks, so rgpot_repair_wheel.sh
# ships librgpot.so.3 and librgpot.so as copies of librgpot.so.X.Y.Z.
# auditwheel then rewrites NEEDED (libopenblas.so.0 -> the vendored,
# hash-named copy under rgpot.libs) in librgpot.so.X.Y.Z alone. _core
# loads librgpot.so.3, which still names libopenblas.so.0, so
# `import rgpot` fails wherever the host has no system openblas. This
# copies the repaired file over both names and rewrites RECORD.
set -euo pipefail
WHL="${1:?wheel path}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 -m zipfile -e "$WHL" "$WORK"
LIBS=$(find "$WORK" -type d -name '.rgpot.mesonpy.libs' -print -quit)
[[ -n "$LIBS" ]] || { echo "no .rgpot.mesonpy.libs in $WHL" >&2; exit 1; }
real=$(find "$LIBS" -maxdepth 1 -type f -name 'librgpot.so.*.*.*' | head -1)
[[ -n "$real" ]] || { echo "no librgpot.so.X.Y.Z in $WHL" >&2; exit 1; }
for name in librgpot.so.3 librgpot.so; do
  cp -f "$real" "$LIBS/$name"
done
echo "soname copies refreshed from $(basename "$real")"
python3 - "$WORK" "$WHL" <<'PY'
import base64
import hashlib
import sys
import zipfile
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
record = next(root.glob("*.dist-info/RECORD"))
lines = []
for p in sorted(root.rglob("*")):
    if p.is_file() and p != record:
        data = p.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        lines.append(f"{p.relative_to(root).as_posix()},sha256={digest.decode()},{len(data)}")
lines.append(f"{record.relative_to(root).as_posix()},,")
record.write_text("\n".join(lines) + "\n")
tmp = out.with_suffix(".whl.tmp")
with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(root).as_posix())
tmp.replace(out)
print("SONAME_SYNC_OK", out)
PY
