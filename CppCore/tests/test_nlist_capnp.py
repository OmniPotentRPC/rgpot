#!/usr/bin/env python3
"""Round-trip a neighbour request and one chunk through Potentials.capnp.

The messages are additive fields. An older client that never sets them still
decodes; this test checks that a writer and a reader of the current schema
agree on one stored chunk and on a stream chunk that says more follows.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run(capnp: str, mode: str, schema: Path, type_name: str, text: str | None = None,
        data: bytes | None = None) -> bytes:
    cmd = [capnp, mode, "--packed", str(schema), type_name]
    if mode == "decode":
        cmd.insert(2, "--short")
    return subprocess.run(
        cmd,
        input=data if data is not None else text.encode(),
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    capnp = sys.argv[2] if len(sys.argv) > 2 else shutil.which("capnp")
    if capnp is None:
        raise SystemExit("capnp is required")
    schema = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    request = (
        "(cutoff = 3.5, skin = 0.2, format = 1, cut = 0, "
        "periodic = [true, true, false], shifts = true, vectors = true, "
        "distances = false, mode = 2)"
    )
    encoded = run(capnp, "encode", schema, "NeighborRequest", request)
    decoded = run(capnp, "decode", schema, "NeighborRequest", data=encoded).decode()
    for piece in ("cutoff = 3.5", "format = 1", "mode = 2", "periodic = [true, true, false]"):
        if piece not in decoded:
            raise SystemExit(f"request lost {piece}: {decoded}")
    chunk = (
        "(first = [0, 1], second = [1, 0], shift = [0, 0, 1, 0, 0, -1], "
        "vector = [1, 0, 0, -1, 0, 0], distance = [1, 1], mask = [255, 255], more = true)"
    )
    encoded_chunk = run(capnp, "encode", schema, "NeighborList", chunk)
    decoded_chunk = run(capnp, "decode", schema, "NeighborList", data=encoded_chunk).decode()
    for piece in ("first = [0, 1]", "more = true", "shift = [0, 0, 1, 0, 0, -1]"):
        if piece not in decoded_chunk:
            raise SystemExit(f"chunk lost {piece}: {decoded_chunk}")
    # A ForceInput that does not mention neighbors still encodes.
    bare = "(pos = [0, 0, 0], atmnrs = [1], box = [1, 0, 0, 0, 1, 0, 0, 0, 1])"
    encoded_bare = run(capnp, "encode", schema, "ForceInput", bare)
    decoded_bare = run(capnp, "decode", schema, "ForceInput", data=encoded_bare).decode()
    if "neighbors =" in decoded_bare:
        raise SystemExit(f"unset neighbours leaked into an old input: {decoded_bare}")
    print("nlist capnp round-trip ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
