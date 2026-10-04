#!/usr/bin/env python3
"""Decode a Capabilities message across the schema that predates build identity.

argv: <repo root> <capnp>

The fixture is the canonical schema at e825c207886b336b4b9c1197e968f184c1b2eb47
(sha256 252f26995fbfa1dccc236e8adec58bd5db2ff27e541d2ac7b80c200d04147039).
It has no ``buildVersion``, ``buildRevision``, or ``getCapabilities``.
The new schema is the vendored ``Potentials.capnp`` this tree compiles.
An old message decodes on the new schema with both text fields absent.
A new message with both fields set decodes on the old schema, and the old
reader does not surface those fields.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

FIXTURE_SHA256 = "252f26995fbfa1dccc236e8adec58bd5db2ff27e541d2ac7b80c200d04147039"

OLD_MESSAGE = """(
  backendName = "probe",
  protocolFamily = "rgpot.potentials",
  protocolMajor = 1,
  protocolMinor = 0,
  schemaId = "0xbd1f89fa17369103",
  bridgeAbiMajor = 1,
  bridgeAbiMinor = 0,
  bridgeLayout = 1,
  dlpackMajor = 1,
  dlpackMinor = 0,
  bridgeFeatures = 3,
  operations = [energy, forces]
)
"""

NEW_MESSAGE = """(
  backendName = "probe",
  protocolFamily = "rgpot.potentials",
  protocolMajor = 1,
  protocolMinor = 0,
  schemaId = "0xbd1f89fa17369103",
  bridgeAbiMajor = 1,
  bridgeAbiMinor = 0,
  bridgeLayout = 1,
  dlpackMajor = 1,
  dlpackMinor = 0,
  bridgeFeatures = 3,
  operations = [energy, forces],
  buildVersion = "1.15.1",
  buildRevision = "79c95ab49748"
)
"""

KEPT = (
    'backendName = "probe"',
    'protocolFamily = "rgpot.potentials"',
    'protocolMajor = 1',
    'protocolMinor = 0',
    'schemaId = "0xbd1f89fa17369103"',
    "bridgeAbiMajor = 1",
    "bridgeAbiMinor = 0",
    "bridgeLayout = 1",
    "dlpackMajor = 1",
    "dlpackMinor = 0",
    "bridgeFeatures = 3",
    "operations = [energy, forces]",
)


def _run(capnp: str, args: list[str], data: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([capnp, *args], input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _encode(capnp: str, schema: Path, text: str) -> bytes:
    proc = _run(capnp, ["encode", str(schema), "Capabilities"], text.encode())
    if proc.returncode != 0:
        raise AssertionError(
            f"capnp encode {schema.name} failed:\n{proc.stderr.decode(errors='replace')}"
        )
    if not proc.stdout:
        raise AssertionError(f"capnp encode {schema.name} wrote an empty message")
    return proc.stdout


def _decode(capnp: str, schema: Path, message: bytes) -> str:
    proc = _run(capnp, ["decode", str(schema), "Capabilities"], message)
    if proc.returncode != 0:
        raise AssertionError(
            f"capnp decode {schema.name} failed:\n{proc.stderr.decode(errors='replace')}"
        )
    return proc.stdout.decode()


def _require_kept(text: str, label: str) -> None:
    for line in KEPT:
        if line not in text:
            raise AssertionError(f"{label} dropped {line!r}:\n{text}")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    capnp = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else "capnp"
    old = root / "CppCore" / "tests" / "fixtures" / "Potentials.pre-build-identity.capnp"
    new = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    rust = root / "rgpot-core" / "schema" / "Potentials.capnp"

    fixture = old.read_bytes()
    digest = hashlib.sha256(fixture).hexdigest()
    if digest != FIXTURE_SHA256:
        raise AssertionError(f"fixture sha256 {digest} != {FIXTURE_SHA256}")
    if b"buildVersion" in fixture or b"buildRevision" in fixture or b"getCapabilities @12" in fixture:
        raise AssertionError("pre-build-identity fixture must not carry the new fields or method")

    vendored = new.read_bytes()
    if vendored != rust.read_bytes():
        raise AssertionError("vendored Potentials.capnp copies differ")
    if b"buildVersion   @18 :Text" not in vendored or b"buildRevision  @19 :Text" not in vendored:
        raise AssertionError("vendored schema must append buildVersion @18 and buildRevision @19")
    if b"getCapabilities @12" not in vendored:
        raise AssertionError("vendored schema must declare getCapabilities @12")

    rejected = _run(capnp, ["encode", str(old), "Capabilities"], NEW_MESSAGE.encode())
    if rejected.returncode == 0:
        raise AssertionError("the old schema accepted buildVersion and buildRevision")

    old_message = _encode(capnp, old, OLD_MESSAGE)
    old_on_new = _decode(capnp, new, old_message)
    _require_kept(old_on_new, "new reader of an old message")
    if "buildVersion" in old_on_new or "buildRevision" in old_on_new:
        raise AssertionError(f"new reader surfaced an empty build field:\n{old_on_new}")

    new_message = _encode(capnp, new, NEW_MESSAGE)
    new_on_old = _decode(capnp, old, new_message)
    _require_kept(new_on_old, "old reader of a new message")
    for absent in ("buildVersion", "buildRevision", "1.15.1", "79c95ab49748"):
        if absent in new_on_old:
            raise AssertionError(f"old reader surfaced {absent!r}:\n{new_on_old}")

    new_on_new = _decode(capnp, new, new_message)
    _require_kept(new_on_new, "new reader of a new message")
    if 'buildVersion = "1.15.1"' not in new_on_new:
        raise AssertionError(f"new reader lost buildVersion:\n{new_on_new}")
    if 'buildRevision = "79c95ab49748"' not in new_on_new:
        raise AssertionError(f"new reader lost buildRevision:\n{new_on_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
