#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Call libuma_engine.so and surface the C++ exception text.

A nonzero return used to be reported as "uma force returned 2". The
engine already stores the exception on the handle. This harness prints
that text and raises it.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path


def encode_uma_params(model: str, schema: Path) -> bytes:
    text = (
        f'(modelPath = "{model}", taskName = "omol", device = "cpu", '
        f"charge = 0, spin = 1)"
    )
    proc = subprocess.run(
        ["capnp", "encode", str(schema), "UmaParams"],
        input=text.encode(),
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode() or "capnp encode failed")
    return proc.stdout


class UmaEngine:
    def __init__(self, library: str, model: str, schema: Path):
        self.lib = ctypes.CDLL(library)
        self.lib.rgpot_engine_create.restype = ctypes.c_void_p
        self.lib.rgpot_engine_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.rgpot_engine_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rgpot_engine_force.restype = ctypes.c_int
        self.lib.rgpot_engine_force.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.rgpot_engine_last_error.restype = ctypes.c_size_t
        self.lib.rgpot_engine_last_error.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        blob = encode_uma_params(model, schema)
        err = ctypes.create_string_buffer(1024)
        self.pot = self.lib.rgpot_engine_create(
            blob, len(blob), err, len(err)
        )
        if not self.pot:
            raise RuntimeError(err.value.decode() or "rgpot_engine_create failed")

    def last_error(self) -> str:
        needed = self.lib.rgpot_engine_last_error(self.pot, None, 0)
        buf = ctypes.create_string_buffer(max(needed + 1, 1))
        self.lib.rgpot_engine_last_error(self.pot, buf, len(buf))
        return buf.value.decode()

    def force(self, pos: list[float], atomic_numbers: list[int], box: list[float]):
        n = len(atomic_numbers)
        c_pos = (ctypes.c_double * (3 * n))(*pos)
        c_z = (ctypes.c_int * n)(*atomic_numbers)
        c_f = (ctypes.c_double * (3 * n))()
        c_box = (ctypes.c_double * 9)(*box)
        energy = ctypes.c_double()
        rc = self.lib.rgpot_engine_force(
            self.pot,
            n,
            c_pos,
            c_z,
            c_f,
            ctypes.byref(energy),
            None,
            c_box,
            None,
            None,
        )
        if rc != 0:
            message = self.last_error()
            if not message:
                message = f"uma force returned {rc}"
            print(message, file=sys.stderr)
            raise RuntimeError(message)
        return energy.value, list(c_f)

    def close(self) -> None:
        if self.pot:
            self.lib.rgpot_engine_destroy(self.pot)
            self.pot = None


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: uma_engine.py LIBUMA.so MODEL.pt2 SCHEMA.capnp",
            file=sys.stderr,
        )
        return 2
    engine = UmaEngine(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
    # s-tetrazine with one hydrogen moved 12 angstroms. The compiled
    # list is the complete graph, 8*7 edges, so the call returns the
    # gather energy (n-1) times the coordinate sum.
    pos = [
        12.5, 13.83, 12.5, 12.5, 11.17, 12.5, 13.65, 13.165, 12.5,
        13.65, 11.835, 12.5, 11.35, 13.165, 12.5, 11.35, 11.835, 12.5,
        12.5, 14.91, 12.5, 12.5, 22.09, 12.5,
    ]
    z = [6, 6, 7, 7, 7, 7, 1, 1]
    box = [40.0, 0, 0, 0, 40.0, 0, 0, 0, 40.0]
    expect = (len(z) - 1) * sum(pos)
    try:
        energy, _forces = engine.force(pos, z, box)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 2
    finally:
        engine.close()
    if abs(energy - expect) > 1e-6:
        print(f"energy {energy} wanted {expect}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
