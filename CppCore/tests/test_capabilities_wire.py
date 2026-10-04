#!/usr/bin/env python3
"""Cross-version wire check for Potentials.capnp Capabilities.

A message written before buildVersion/buildRevision (@18, @19) existed must
decode under the current schema with both fields empty, and a message that
carries them must decode under the older definition with every earlier field
intact. Usage: test_capabilities_wire.py <root> [<capnp>]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = (
    'backendName = "cpmdc", backendVersion = "1.2", abiVersion = 3, '
    'available = true, operations = [energy, forces], '
    'protocolFamily = "rgpot.potentials", protocolMajor = 1, protocolMinor = 0, '
    'schemaId = "0xbd1f89fa17369103", bridgeAbiMajor = 1, bridgeLayout = 1, '
    "dlpackMajor = 1, bridgeFeatures = 3"
)
BUILD = 'buildVersion = "3.5.0", buildRevision = "abc123def456"'


def run(capnp: str, mode: str, schema: Path, text: str | None = None, data: bytes | None = None) -> bytes:
    cmd = [capnp, mode, "--packed", str(schema), "Capabilities"]
    if mode == "decode":
        cmd.insert(2, "--short")
    out = subprocess.run(
        cmd,
        input=data if data is not None else text.encode(),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return out


def fields(decoded: str) -> dict[str, str]:
    inner = decoded.strip()[1:-1]
    parts = re.split(r",\s*(?=[a-zA-Z]+ = )", inner)
    return dict(p.split(" = ", 1) for p in parts)


def main() -> int:
    root = Path(sys.argv[1])
    capnp = sys.argv[2] if len(sys.argv) > 2 else shutil.which("capnp")
    if capnp is None:
        raise AssertionError("capnp is required")
    new = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    old = root / "CppCore" / "tests" / "data" / "capabilities_before_build_identity.capnp"

    schema_text = new.read_text(encoding="utf-8")
    for ordinal, name in ((18, "buildVersion"), (19, "buildRevision")):
        if not re.search(rf"{name}\s+@{ordinal}\s*:\s*Text", schema_text):
            raise AssertionError(f"{name} must be Text @{ordinal} (append only)")

    # New fields written, read by the older definition.
    written_new = run(capnp, "encode", new, text=f"({BASE}, {BUILD})")
    seen_by_old = fields(run(capnp, "decode", old, data=written_new).decode())
    expected = fields(f"({BASE})")
    for key in ("backendName", "protocolFamily", "protocolMajor", "bridgeFeatures", "schemaId"):
        assert seen_by_old[key] == expected[key], (key, seen_by_old[key])
    assert "buildVersion" not in seen_by_old

    # Older message, read by the current definition.
    written_old = run(capnp, "encode", old, text=f"({BASE})")
    seen_by_new = fields(run(capnp, "decode", new, data=written_old).decode())
    for key in ("backendName", "protocolFamily", "protocolMajor", "bridgeFeatures", "schemaId"):
        assert seen_by_new[key] == expected[key], (key, seen_by_new[key])
    assert seen_by_new.get("buildVersion", '""') == '""', seen_by_new
    assert seen_by_new.get("buildRevision", '""') == '""', seen_by_new

    # Round trip under the current definition keeps the new fields.
    round_trip = fields(run(capnp, "decode", new, data=written_new).decode())
    assert round_trip["buildVersion"] == '"3.5.0"', round_trip
    assert round_trip["buildRevision"] == '"abc123def456"', round_trip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
