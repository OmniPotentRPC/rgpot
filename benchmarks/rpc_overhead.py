"""Per-call LJ round trip through potserv.

asv times ``time_round_trip`` only. ``setup`` starts one server and keeps
the client connection. The row is skipped when ``RGPOT_POTSERV`` or
``RGPOT_POT_BRIDGE`` is unset or does not name a file, which is the
default wheel (``with_rpc`` off).
"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import time


def _fcc(n: int, lattice: float = 4.05):
    import numpy as np

    basis = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    ncell = 8
    coords = []
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                for site in basis:
                    if len(coords) >= n:
                        break
                    coords.append(
                        (
                            (ix + site[0]) * lattice,
                            (iy + site[1]) * lattice,
                            (iz + site[2]) * lattice,
                        )
                    )
                if len(coords) >= n:
                    break
            if len(coords) >= n:
                break
        if len(coords) >= n:
            break
    pos = np.ascontiguousarray(coords, dtype=np.float64).ravel()
    types = np.full(n, 18, dtype=np.int32)
    box = np.zeros(9, dtype=np.float64)
    edge = ncell * lattice
    box[0] = box[4] = box[8] = edge
    return pos, types, box


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _load_bridge(path: str):
    lib = ctypes.CDLL(path)
    lib.pot_client_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.pot_client_init.restype = ctypes.c_void_p
    lib.pot_client_free.argtypes = [ctypes.c_void_p]
    lib.pot_client_free.restype = None
    lib.pot_get_last_error.argtypes = [ctypes.c_void_p]
    lib.pot_get_last_error.restype = ctypes.c_char_p
    lib.pot_calculate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.pot_calculate.restype = ctypes.c_int32
    return lib


class RpcLjRoundTrip:
    """One warm LJ force call through a live potserv connection."""

    params = [7, 256, 1024]
    param_names = ["atoms"]
    rounds = 8
    repeat = 1
    sample_time = 0.05
    warmup_time = 0.05
    min_run_count = 1
    timeout = 180

    def setup(self, atoms: int) -> None:
        server = os.environ.get("RGPOT_POTSERV", "")
        bridge = os.environ.get("RGPOT_POT_BRIDGE", "")
        if not server or not bridge or not os.path.isfile(server) or not os.path.isfile(bridge):
            raise NotImplementedError("potserv is not available")
        self.pos, self.types, self.box = _fcc(atoms)
        self.energy = ctypes.c_double(0.0)
        self.forces = (ctypes.c_double * (atoms * 3))()
        self.port = _free_port()
        self.proc = subprocess.Popen(
            [server, str(self.port), "LJ"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.lib = _load_bridge(bridge)
        self.client = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("potserv exited before accepting a client")
            try:
                with socket.create_connection(("127.0.0.1", self.port), 0.05):
                    pass
            except OSError:
                time.sleep(0.02)
                continue
            client = self.lib.pot_client_init(b"127.0.0.1", self.port)
            if client:
                self.client = client
                break
        if not self.client:
            self.proc.terminate()
            raise RuntimeError("potserv did not accept a client")
        self._call()

    def _call(self) -> None:
        rc = self.lib.pot_calculate(
            self.client,
            self.types.size,
            self.pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self.types.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self.box.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(self.energy),
            self.forces,
        )
        if rc != 0:
            err = self.lib.pot_get_last_error(self.client) or b""
            raise RuntimeError(err.decode())

    def time_round_trip(self, atoms: int) -> None:
        del atoms
        self._call()

    def teardown(self, atoms: int) -> None:
        del atoms
        client = getattr(self, "client", None)
        lib = getattr(self, "lib", None)
        if client and lib:
            lib.pot_client_free(client)
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
