# Copyright 2023--present rgpot developers
"""Persistent dxtb engine checks against dxtb 0.4.0 autograd."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

import capnp
import dxtb
import numpy as np
import pytest
import torch

ANGSTROM_TO_BOHR = 1.8897261254578281
HARTREE_TO_EV = 27.211386245988
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = capnp.load(str(ROOT / "CppCore/rgpot/rpc/Potentials.capnp"))
BOX = np.diag([40.0, 40.0, 40.0]).astype(np.float64)


class AbiStamp(ctypes.Structure):
    _fields_ = [
        ("abi_major", ctypes.c_ushort),
        ("abi_minor", ctypes.c_ushort),
        ("layout_revision", ctypes.c_ushort),
    ]


class Caps(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("reentrancy", ctypes.c_int),
        ("per_image_instances", ctypes.c_int),
        ("batched", ctypes.c_int),
        ("periodic", ctypes.c_int),
        ("stress", ctypes.c_int),
        ("group_collective", ctypes.c_int),
        ("orbital_selection", ctypes.c_int),
        ("orbital_copy", ctypes.c_int),
        ("analytic_hvp", ctypes.c_int),
    ]


def expected_abi_stamp() -> tuple[int, int, int]:
    header = (ROOT / "CppCore/rgpot/engine_c_abi.h").read_text()

    def macro(name: str) -> int:
        match = re.search(rf"^#define\s+{name}\s+(\d+)\b", header, re.MULTILINE)
        assert match is not None, f"{name} missing from engine_c_abi.h"
        return int(match.group(1))

    return (
        macro("RGPOT_ENGINE_ABI_VERSION"),
        macro("RGPOT_ENGINE_ABI_MINOR"),
        macro("RGPOT_ENGINE_ABI_LAYOUT_REVISION"),
    )


class Engine:
    def __init__(
        self,
        path: str,
        name: str,
        method: str = "gfn2",
        charge: float = 0.0,
        uhf: int = 0,
    ):
        self.lib = ctypes.CDLL(path)
        self.lib.rgpot_engine_abi_version.restype = ctypes.c_int
        self.lib.rgpot_engine_abi_version.argtypes = []
        self.lib.rgpot_engine_abi_stamp.restype = AbiStamp
        self.lib.rgpot_engine_abi_stamp.argtypes = []
        self.lib.rgpot_engine_abi_compatible.restype = ctypes.c_int
        self.lib.rgpot_engine_abi_compatible.argtypes = [ctypes.POINTER(AbiStamp)]
        self.lib.rgpot_engine_name.restype = ctypes.c_char_p
        self.lib.rgpot_engine_name.argtypes = []
        self.lib.rgpot_engine_version.restype = ctypes.c_char_p
        self.lib.rgpot_engine_version.argtypes = []
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
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.rgpot_engine_hvp.restype = ctypes.c_int
        self.lib.rgpot_engine_hvp.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.rgpot_engine_caps.restype = ctypes.c_int
        self.lib.rgpot_engine_caps.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Caps),
        ]
        self.lib.rgpot_engine_set_num_threads.restype = ctypes.c_int
        self.lib.rgpot_engine_set_num_threads.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.rgpot_engine_last_error.restype = ctypes.c_size_t
        self.lib.rgpot_engine_last_error.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        message = SCHEMA.BuiltinParams.new_message()
        message.name = name
        config = message.init("config")
        xtb = config.init("xtb")
        xtb.method = method
        xtb.accuracy = 1.0
        xtb.electronicTemperature = 300.0
        xtb.maxIterations = 300
        xtb.charge = charge
        xtb.uhf = uhf
        encoded = message.to_bytes()
        storage = ctypes.create_string_buffer(encoded)
        error = ctypes.create_string_buffer(1024)
        self.handle = self.lib.rgpot_engine_create(
            storage, len(encoded), error, len(error)
        )
        if not self.handle:
            raise RuntimeError(error.value.decode())

    def close(self):
        if self.handle:
            self.lib.rgpot_engine_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def caps(self) -> Caps:
        caps = Caps()
        caps.size = ctypes.sizeof(caps)
        assert self.lib.rgpot_engine_caps(self.handle, ctypes.byref(caps)) == 0
        return caps

    def abi_version(self) -> int:
        return int(self.lib.rgpot_engine_abi_version())

    def abi_stamp(self) -> AbiStamp:
        return self.lib.rgpot_engine_abi_stamp()

    def abi_compatible(self, host: AbiStamp | None) -> int:
        if host is None:
            return int(self.lib.rgpot_engine_abi_compatible(None))
        return int(self.lib.rgpot_engine_abi_compatible(ctypes.byref(host)))

    def engine_name(self) -> str:
        return self.lib.rgpot_engine_name().decode()

    def engine_version(self) -> str:
        return self.lib.rgpot_engine_version().decode()

    def set_num_threads(self, intra_op: int, inter_op: int) -> int:
        return int(
            self.lib.rgpot_engine_set_num_threads(self.handle, intra_op, inter_op)
        )

    def last_error(self) -> str:
        size = int(self.lib.rgpot_engine_last_error(self.handle, None, 0))
        if size == 0:
            return ""
        buf = ctypes.create_string_buffer(size + 1)
        self.lib.rgpot_engine_last_error(self.handle, buf, len(buf))
        return buf.value.decode()

    @staticmethod
    def _arrays(numbers, positions):
        z = np.ascontiguousarray(numbers, dtype=np.int32)
        x = np.ascontiguousarray(positions, dtype=np.float64).reshape(-1)
        return z, x

    def force(self, numbers, positions):
        z, x = self._arrays(numbers, positions)
        forces = np.empty_like(x)
        energy = ctypes.c_double()
        status = self._force_call(z, x, forces, energy)
        assert status == 0
        return energy.value, forces

    def force_status(self, numbers, positions):
        z, x = self._arrays(numbers, positions)
        forces = np.empty_like(x)
        energy = ctypes.c_double()
        return self._force_call(z, x, forces, energy)

    def _force_call(self, z, x, forces, energy):
        return self.lib.rgpot_engine_force(
            self.handle,
            z.size,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(energy),
            None,
            BOX.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            None,
            None,
        )

    def hvp(self, numbers, positions, direction):
        z, x = self._arrays(numbers, positions)
        v = np.ascontiguousarray(direction, dtype=np.float64).reshape(-1)
        product = np.empty_like(v)
        status = self.lib.rgpot_engine_hvp(
            self.handle,
            z.size,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            product.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            BOX.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        assert status == 0
        return product

    def aliased_hvp_status(self, numbers, positions, direction):
        z, x = self._arrays(numbers, positions)
        v = np.ascontiguousarray(direction, dtype=np.float64).reshape(-1)
        ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self.lib.rgpot_engine_hvp(
            self.handle,
            z.size,
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ptr,
            ptr,
            BOX.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )


def direct_calculator(numbers, method):
    torch.set_num_threads(1)
    z = torch.tensor(numbers, dtype=torch.int64)
    par = dxtb.GFN1_XTB if method == "gfn1" else dxtb.GFN2_XTB
    opts = {
        "scf_mode": "full",
        "int_driver": "libcint",
        "f_atol": 1.0e-11,
        "x_atol": 1.0e-11,
        "maxiter": 300,
        "mixer": "simple",
        "damp": 0.3,
        "force_convergence": True,
        "fermi_etemp": 300.0,
        "cache_enabled": False,
        "verbosity": 0,
    }
    return dxtb.Calculator(z, par, opts=opts, dtype=torch.float64)


def direct_hessian(numbers, positions, method):
    calc = direct_calculator(numbers, method)
    calc.reset()
    x = (
        torch.tensor(positions, dtype=torch.float64) * ANGSTROM_TO_BOHR
    ).requires_grad_(True)
    energy = calc.energy(x, chrg=0.0, spin=0)
    gradient = torch.autograd.grad(energy.sum(), x, create_graph=True)[0].reshape(-1)
    rows = []
    for index, value in enumerate(gradient):
        row = torch.autograd.grad(
            value, x, retain_graph=index + 1 < gradient.numel()
        )[0]
        rows.append(row.reshape(-1))
    hessian = torch.stack(rows).detach().numpy()
    return hessian * HARTREE_TO_EV * ANGSTROM_TO_BOHR**2


def direct_force(numbers, positions, method, charge, uhf):
    calc = direct_calculator(numbers, method)
    calc.reset()
    x = (
        torch.tensor(positions, dtype=torch.float64) * ANGSTROM_TO_BOHR
    ).requires_grad_(True)
    energy = calc.energy(x, chrg=charge, spin=uhf)
    gradient = torch.autograd.grad(energy.sum(), x)[0]
    force = -gradient.detach().numpy() * HARTREE_TO_EV * ANGSTROM_TO_BOHR
    return float(energy.detach()) * HARTREE_TO_EV, force.reshape(-1)


MOLECULES = [
    (
        "h2",
        "gfn1",
        [1, 1],
        [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
    ),
    (
        "water",
        "gfn1",
        [8, 1, 1],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    ),
    (
        "methane",
        "gfn2",
        [6, 1, 1, 1, 1],
        [
            [0.0, 0.0, 0.0],
            [0.6291, 0.6291, 0.6291],
            [-0.6291, -0.6291, 0.6291],
            [-0.6291, 0.6291, -0.6291],
            [0.6291, -0.6291, -0.6291],
        ],
    ),
]


@pytest.mark.parametrize("_name,method,numbers,positions", MOLECULES)
def test_hvp_matches_dense_autograd(_name, method, numbers, positions):
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        positions = np.asarray(positions)
        direction = np.linspace(-0.7, 0.9, positions.size).reshape(positions.shape)
        product = engine.hvp(numbers, positions, direction)
        dense = direct_hessian(numbers, positions, method)
        np.testing.assert_allclose(
            product, dense @ direction.reshape(-1), rtol=5.0e-10, atol=5.0e-9
        )
        caps = engine.caps()
        assert caps.analytic_hvp == 1
        assert caps.periodic == 0
        assert caps.reentrancy == 1
    finally:
        engine.close()


def test_hessian_products_are_symmetric():
    _, method, numbers, positions = MOLECULES[1]
    positions = np.asarray(positions)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        v = np.linspace(-0.4, 0.8, positions.size)
        w = np.linspace(0.9, -0.2, positions.size)
        hv = engine.hvp(numbers, positions, v)
        hw = engine.hvp(numbers, positions, w)
        np.testing.assert_allclose(np.dot(hv, w), np.dot(v, hw), rtol=2e-11, atol=2e-10)
        assert engine.aliased_hvp_status(numbers, positions, v) == 1
        assert engine.force_status([0], [[0.0, 0.0, 0.0]]) == 1
    finally:
        engine.close()


def test_charge_and_spin_come_from_config():
    numbers = [1, 1]
    positions = [[0.0, 0.0, -0.45], [0.0, 0.0, 0.45]]
    engine = Engine(
        os.environ["RGPOT_ENGINE_DXTB"],
        "dxtb",
        "gfn1",
        charge=1.0,
        uhf=1,
    )
    try:
        energy, forces = engine.force(numbers, positions)
        ref_energy, ref_forces = direct_force(
            numbers, positions, "gfn1", charge=1.0, uhf=1
        )
        np.testing.assert_allclose(energy, ref_energy, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(forces, ref_forces, rtol=2e-12, atol=2e-11)
    finally:
        engine.close()


def read_con(path):
    lines = path.read_text().splitlines()
    counts = [int(value) for value in lines[7].split()]
    numbers = []
    positions = []
    atomic_number = {"H": 1, "C": 6, "O": 8}
    row = 9
    for count in counts:
        symbol = lines[row]
        row += 2
        for _ in range(count):
            positions.append([float(value) for value in lines[row].split()[:3]])
            numbers.append(atomic_number[symbol])
            row += 1
    return numbers, np.asarray(positions)


@pytest.mark.parametrize("stationary", ["minimum", "saddle"])
def test_malonaldehyde_hvp_matches_dense_autograd(stationary):
    path = ROOT / f"CppCore/tests/fixtures/malonaldehyde_{stationary}.con"
    numbers, positions = read_con(path)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn2")
    try:
        direction = np.linspace(-0.5, 0.75, positions.size)
        product = engine.hvp(numbers, positions, direction)
        dense = direct_hessian(numbers, positions, "gfn2")
        np.testing.assert_allclose(
            product, dense @ direction, rtol=8.0e-10, atol=2.0e-9
        )
    finally:
        engine.close()


@pytest.mark.skipif("RGPOT_ENGINE_XTB" not in os.environ, reason="libxtb engine not built")
@pytest.mark.parametrize("_name,_method,numbers,positions", MOLECULES[1:])
def test_energy_force_parity_with_libxtb(_name, _method, numbers, positions):
    dxtb_engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn2")
    xtb_engine = Engine(os.environ["RGPOT_ENGINE_XTB"], "xtb", "gfn2")
    try:
        de, df = dxtb_engine.force(numbers, positions)
        xe, xf = xtb_engine.force(numbers, positions)
        report = {
            "molecule": _name,
            "abs_energy_eV": abs(de - xe),
            "max_abs_force_eV_per_A": float(np.max(np.abs(df - xf))),
        }
        print(json.dumps(report, sort_keys=True))
        assert report["abs_energy_eV"] < 5.0e-3
        assert report["max_abs_force_eV_per_A"] < 5.0e-3
    finally:
        dxtb_engine.close()
        xtb_engine.close()


def test_hvp_force_cost_report():
    _, method, numbers, positions = MOLECULES[1]
    positions = np.asarray(positions)
    direction = np.linspace(-0.5, 0.5, positions.size)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        engine.force(numbers, positions)
        engine.hvp(numbers, positions, direction)
        start = time.perf_counter()
        engine.force(numbers, positions)
        force_seconds = time.perf_counter() - start
        start = time.perf_counter()
        engine.hvp(numbers, positions, direction)
        hvp_seconds = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "force_seconds": force_seconds,
                    "hvp_seconds": hvp_seconds,
                    "hvp_over_force": hvp_seconds / force_seconds,
                },
                sort_keys=True,
            )
        )
        assert force_seconds > 0.0
        assert hvp_seconds > 0.0
    finally:
        engine.close()


def test_abi_identity_and_stamp():
    expected_major, expected_minor, expected_layout = expected_abi_stamp()
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn2")
    try:
        assert engine.abi_version() == expected_major
        assert engine.engine_name() == "dxtb"
        assert engine.engine_version().startswith("dxtb/")
        stamp = engine.abi_stamp()
        assert (stamp.abi_major, stamp.abi_minor, stamp.layout_revision) == (
            expected_major,
            expected_minor,
            expected_layout,
        )
        assert stamp.abi_minor > 0
        assert engine.abi_compatible(AbiStamp(expected_major, expected_minor, expected_layout)) == 1
        assert engine.abi_compatible(AbiStamp(expected_major, expected_minor + 1, expected_layout)) == 1
        assert engine.abi_compatible(AbiStamp(expected_major, expected_minor - 1, expected_layout)) == 0
        assert engine.abi_compatible(AbiStamp(expected_major + 1, expected_minor, expected_layout)) == 0
        assert engine.abi_compatible(AbiStamp(expected_major, expected_minor, expected_layout + 1)) == 0
        assert engine.abi_compatible(None) == 0
    finally:
        engine.close()


def test_set_num_threads_idempotent_and_restorable():
    _, method, numbers, positions = MOLECULES[0]
    positions = np.asarray(positions)
    ref_energy, ref_forces = direct_force(numbers, positions, method, charge=0.0, uhf=0)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        assert engine.set_num_threads(0, 0) == 0
        assert engine.set_num_threads(-1, -1) == 0
        assert engine.set_num_threads(2, 1) == 0
        assert engine.last_error() == ""
        assert engine.set_num_threads(2, 1) == 0
        assert engine.last_error() == ""

        energy, forces = engine.force(numbers, positions)
        np.testing.assert_allclose(energy, ref_energy, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(forces, ref_forces, rtol=2e-12, atol=2e-11)

        assert engine.set_num_threads(2, 1) == 0
        assert engine.last_error() == ""
        assert engine.set_num_threads(3, -1) == 0
        assert engine.last_error() == ""
        assert engine.set_num_threads(2, -1) == 0
        assert engine.last_error() == ""

        # Changing the inter-op count once locked must be refused with code 2.
        if engine.set_num_threads(-1, 2) == 0:
            assert engine.set_num_threads(-1, 3) == 2
        else:
            assert engine.set_num_threads(-1, 2) == 2
        assert engine.last_error() != ""

        # Restoring the active counts succeeds and clears the error.
        assert engine.set_num_threads(2, -1) == 0
        assert engine.last_error() == ""
        energy2, forces2 = engine.force(numbers, positions)
        np.testing.assert_allclose(energy2, ref_energy, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(forces2, ref_forces, rtol=2e-12, atol=2e-11)
    finally:
        engine.close()


def test_set_num_threads_worker_has_no_redundant_calls_or_reinit(tmp_path, monkeypatch):
    log_path = tmp_path / "worker_calls.jsonl"
    driver_path = tmp_path / "worker_driver.py"
    wrapper_path = tmp_path / "worker_python.sh"
    real_python = os.environ.get("RGPOT_DXTB_PYTHON") or sys.executable

    driver_path.write_text(
        f"""
import json
import sys
import torch
import dxtb

LOG_PATH = {str(log_path)!r}

def record(event, **extra):
    payload = {{
        "event": event,
        "intra": int(torch.get_num_threads()),
        "inter": int(torch.get_num_interop_threads()),
    }}
    payload.update(extra)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\\n")

_orig_set_num_threads = torch.set_num_threads
def _wrapped_set_num_threads(n):
    _orig_set_num_threads(n)
    record("set_num_threads", arg=int(n))

_orig_set_num_interop_threads = torch.set_num_interop_threads
def _wrapped_set_num_interop_threads(n):
    _orig_set_num_interop_threads(n)
    record("set_num_interop_threads", arg=int(n))

_orig_calculator = dxtb.Calculator
def _wrapped_calculator(*args, **kwargs):
    calc = _orig_calculator(*args, **kwargs)
    record("calculator_init")
    return calc

torch.set_num_threads = _wrapped_set_num_threads
torch.set_num_interop_threads = _wrapped_set_num_interop_threads
dxtb.Calculator = _wrapped_calculator

record("startup")
idx = sys.argv.index("-c")
source = sys.argv[idx + 1]
exec(compile(source, "<dxtb-worker>", "exec"), {{"__name__": "__main__"}})
""".lstrip()
    )
    wrapper_path.write_text(
        f"#!/bin/sh\nexec {shlex.quote(real_python)} {shlex.quote(str(driver_path))} \"$@\"\n"
    )
    wrapper_path.chmod(0o755)
    monkeypatch.setenv("RGPOT_DXTB_PYTHON", str(wrapper_path))

    def read_events():
        return [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    _, method, numbers, positions = MOLECULES[0]
    positions = np.asarray(positions)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        events = read_events()
        startup = [ev for ev in events if ev["event"] == "startup"]
        assert len(startup) == 1
        initial_intra = startup[0]["intra"]
        initial_inter = startup[0]["inter"]

        first_intra = 2 if initial_intra != 2 else 3
        alt_intra = 3 if first_intra == 2 else 2
        first_inter = 1 if initial_inter != 1 else 2
        alt_inter = 2 if first_inter == 1 else 1

        engine.force(numbers, positions)
        engine.force(numbers, positions)
        before_threads = read_events()
        assert sum(1 for ev in before_threads if ev["event"] == "calculator_init") == 1

        assert engine.set_num_threads(first_intra, first_inter) == 0
        after_first = read_events()[len(before_threads):]
        assert [(ev["event"], ev["arg"]) for ev in after_first] == [
            ("set_num_interop_threads", first_inter),
            ("set_num_threads", first_intra),
        ]

        # Calling with the same counts again must be a strict no-op.
        mark = len(read_events())
        assert engine.set_num_threads(first_intra, first_inter) == 0
        assert read_events()[mark:] == []

        # Change intra-op threads, verify no-op when repeated, then restore.
        mark = len(read_events())
        assert engine.set_num_threads(alt_intra, first_inter) == 0
        assert [(ev["event"], ev["arg"]) for ev in read_events()[mark:]] == [
            ("set_num_threads", alt_intra),
        ]

        mark = len(read_events())
        assert engine.set_num_threads(alt_intra, -1) == 0
        assert read_events()[mark:] == []

        mark = len(read_events())
        assert engine.set_num_threads(first_intra, first_inter) == 0
        assert [(ev["event"], ev["arg"]) for ev in read_events()[mark:]] == [
            ("set_num_threads", first_intra),
        ]

        # A refused inter-op change must fail without mutating intra-op threads.
        mark = len(read_events())
        assert engine.set_num_threads(alt_intra, alt_inter) == 2
        assert engine.last_error() != ""
        assert read_events()[mark:] == []

        # Restoring/setting matching counts clears the error and still reuses calculator.
        mark = len(read_events())
        assert engine.set_num_threads(first_intra, first_inter) == 0
        assert engine.last_error() == ""
        assert read_events()[mark:] == []

        engine.force(numbers, positions)
        all_events = read_events()
        assert sum(1 for ev in all_events if ev["event"] == "calculator_init") == 1
        assert all_events[-1]["intra"] == first_intra
        assert all_events[-1]["inter"] == first_inter
    finally:
        engine.close()
