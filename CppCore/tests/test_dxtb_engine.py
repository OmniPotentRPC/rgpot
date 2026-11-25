# Copyright 2023--present rgpot developers
"""Persistent dxtb engine checks against dxtb 0.4.0 autograd."""

from __future__ import annotations

import ast
import ctypes
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
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
        accuracy: float = 1.0,
        electronic_temperature: float = 300.0,
        max_iterations: int = 300,
        library: str = "",
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
        self.lib.rgpot_engine_set_charge_spin.restype = ctypes.c_int
        self.lib.rgpot_engine_set_charge_spin.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.rgpot_engine_set_cache.restype = ctypes.c_int
        self.lib.rgpot_engine_set_cache.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self.lib.rgpot_engine_cache_counts.restype = ctypes.c_int
        self.lib.rgpot_engine_cache_counts.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.rgpot_engine_last_error.restype = ctypes.c_size_t
        self.lib.rgpot_engine_last_error.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        message = SCHEMA.BuiltinParams.new_message()
        message.name = name
        message.library = library
        config = message.init("config")
        xtb = config.init("xtb")
        xtb.method = method
        xtb.accuracy = accuracy
        xtb.electronicTemperature = electronic_temperature
        xtb.maxIterations = max_iterations
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

    def set_charge_spin(self, charge: int, spin: int) -> int:
        return int(self.lib.rgpot_engine_set_charge_spin(self.handle, charge, spin))

    def attach_cache(self, path: Path) -> int:
        return int(self.lib.rgpot_engine_set_cache(self.handle, os.fsencode(path)))

    def cache_counts(self) -> tuple[int, int]:
        computed = ctypes.c_uint64()
        served = ctypes.c_uint64()
        status = self.lib.rgpot_engine_cache_counts(
            self.handle, ctypes.byref(computed), ctypes.byref(served)
        )
        assert status == 0
        return computed.value, served.value

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


def test_cache_separates_every_dxtb_setting_and_runtime_charge_spin(tmp_path):
    numbers = [6]
    positions = [[0.0, 0.0, 0.0]]
    cache = tmp_path / "dxtb-cache"
    variants = [
        {},
        {"method": "gfn2"},
        {"charge": 2.0},
        {"uhf": 2},
        {"accuracy": 0.8},
        {"electronic_temperature": 500.0},
        {"max_iterations": 301},
    ]

    for settings in variants:
        engine = Engine(
            os.environ["RGPOT_ENGINE_DXTB"],
            "dxtb",
            **{"method": "gfn1", **settings},
        )
        try:
            status = engine.attach_cache(cache)
            if status == 3:
                pytest.skip("dxtb engine was built without cache support")
            assert status == 0
            engine.force(numbers, positions)
            assert engine.cache_counts() == (1, 0)
        finally:
            engine.close()

    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn1")
    try:
        assert engine.attach_cache(cache) == 0
        engine.force(numbers, positions)
        assert engine.cache_counts() == (0, 1)
        assert engine.set_charge_spin(2, 1) == 0
        engine.force(numbers, positions)
        assert engine.cache_counts() == (0, 2)
        assert engine.set_charge_spin(0, 3) == 0
        engine.force(numbers, positions)
        assert engine.cache_counts() == (0, 3)
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


FAKE_WORKER = r'''
import json
import os
import signal
import struct
import sys
import time

LOG = os.environ["FAKE_DXTB_LOG"]
REFUSE_UHF = int(os.environ.get("FAKE_DXTB_REFUSE_UHF", "-1"))
MODE = os.environ.get("FAKE_DXTB_MODE", "")
HEADER = struct.Struct("<4sIIII")
CONFIG = struct.Struct("<Iii3dii")
inp = sys.stdin.buffer
out = sys.stdout.buffer


def log(**event):
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")


def reply(status, payload=b""):
    out.write(HEADER.pack(b"DXTR", 1, status & 0xFFFFFFFF, 0, len(payload)))
    out.write(payload)
    out.flush()


def hang():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(60)


def shutdown():
    if MODE == "hang-shutdown":
        log(event="shutdown", result="hang")
        hang()
    if MODE == "linger":
        reply(0)
        log(event="shutdown", result="linger")
        hang()
    if MODE == "drop":
        log(event="shutdown", result="drop")
        os._exit(0)
    if MODE == "partial":
        try:
            out.write(b"DX")
            out.flush()
        except (BrokenPipeError, ConnectionResetError):
            log(event="shutdown", result="broken-pipe")
            os._exit(0)
        log(event="shutdown", result="partial")
        os._exit(0)
    try:
        reply(0)
    except (BrokenPipeError, ConnectionResetError):
        log(event="shutdown", result="broken-pipe")
        os._exit(0)
    if MODE == "late-write":
        # Keep writing after the acknowledgement. An early close on the host
        # turns one of these writes into a broken pipe.
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            try:
                out.write(b"\x00")
                out.flush()
            except (BrokenPipeError, ConnectionResetError):
                log(event="shutdown", result="broken-pipe")
                os._exit(0)
            time.sleep(0.005)
    log(event="shutdown", result="ok")
    os._exit(0)


while True:
    raw = inp.read(HEADER.size)
    if len(raw) != HEADER.size:
        break
    _magic, _version, op, natoms, size = HEADER.unpack(raw)
    payload = inp.read(size)
    if op == 1:
        method, uhf, maxiter, charge, accuracy, temperature, _intra, _inter = (
            CONFIG.unpack(payload)
        )
        log(
            event="configure",
            pid=os.getpid(),
            method=method,
            uhf=uhf,
            maxiter=maxiter,
            charge=charge,
            accuracy=accuracy,
            temperature=temperature,
        )
        if uhf == REFUSE_UHF:
            reply(1, b"ValueError: fake worker refuses this spin")
        else:
            reply(0)
    elif op == 5:
        shutdown()
    elif op == 2 and MODE == "drop-force":
        log(event="force", result="drop")
        os._exit(0)
    else:
        reply(0)
'''


def fake_worker(tmp_path, name, refuse_uhf=-1, mode=""):
    """A python stand-in that logs every configuration the engine sends."""
    log_path = tmp_path / f"{name}.jsonl"
    driver_path = tmp_path / f"{name}_driver.py"
    wrapper_path = tmp_path / f"{name}_python.sh"
    driver_path.write_text(FAKE_WORKER)
    real_python = os.environ.get("RGPOT_DXTB_PYTHON") or sys.executable
    wrapper_path.write_text(
        "#!/bin/sh\n"
        f"FAKE_DXTB_LOG={shlex.quote(str(log_path))} "
        f"FAKE_DXTB_REFUSE_UHF={refuse_uhf} "
        f"FAKE_DXTB_MODE={shlex.quote(mode)} "
        f"exec {shlex.quote(real_python)} {shlex.quote(str(driver_path))} \"$@\"\n"
    )
    wrapper_path.chmod(0o755)
    return wrapper_path, log_path


def configure_events(log_path):
    if not log_path.exists():
        return []
    return [
        event
        for event in (
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if event["event"] == "configure"
    ]


def test_set_charge_spin_refuses_zero_spin_with_code_3():
    _, method, numbers, positions = MOLECULES[0]
    positions = np.asarray(positions)
    ref_energy, ref_forces = direct_force(numbers, positions, method, charge=0.0, uhf=0)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", method)
    try:
        energy, forces = engine.force(numbers, positions)
        np.testing.assert_allclose(energy, ref_energy, rtol=2e-13, atol=2e-12)

        assert engine.set_charge_spin(0, 0) == 3
        assert ">= 1" in engine.last_error()

        # The refused call left the configured charge and spin in force.
        again_energy, again_forces = engine.force(numbers, positions)
        assert again_energy == energy
        np.testing.assert_array_equal(again_forces, forces)
        np.testing.assert_allclose(again_forces, ref_forces, rtol=2e-12, atol=2e-11)

        # A valid change still reaches the worker and clears the error.
        assert engine.set_charge_spin(1, 2) == 0
        assert engine.last_error() == ""
        cation_energy, _ = engine.force(numbers, positions)
        ref_cation, _ = direct_force(numbers, positions, method, charge=1.0, uhf=1)
        np.testing.assert_allclose(cation_energy, ref_cation, rtol=2e-13, atol=2e-12)
    finally:
        engine.close()


def test_set_charge_spin_failure_leaves_the_configuration_unchanged(tmp_path, monkeypatch):
    wrapper, log_path = fake_worker(tmp_path, "refusing", refuse_uhf=2)
    monkeypatch.setenv("RGPOT_DXTB_PYTHON", str(wrapper))
    engine = Engine(
        os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn1", charge=2.0, uhf=0
    )
    try:
        created = configure_events(log_path)
        assert len(created) == 1
        base = created[0]
        assert (base["charge"], base["uhf"]) == (2.0, 0)

        # A refused spin multiplicity is not sent to the worker at all.
        assert engine.set_charge_spin(0, 0) == 3
        assert len(configure_events(log_path)) == 1

        # The worker refuses this configuration; its code and reason come back.
        assert engine.set_charge_spin(1, 3) == 1
        assert "refuses" in engine.last_error()
        refused = configure_events(log_path)[1]
        assert (refused["charge"], refused["uhf"]) == (1.0, 2)

        # The next accepted change carries only the requested charge and spin.
        assert engine.set_charge_spin(-1, 2) == 0
        assert engine.last_error() == ""
        accepted = configure_events(log_path)[2]
        assert (accepted["charge"], accepted["uhf"]) == (-1.0, 1)
        for key in ("method", "maxiter", "accuracy", "temperature"):
            assert accepted[key] == base[key]
    finally:
        engine.close()


def test_library_selects_python_before_the_environment(tmp_path, monkeypatch):
    from_library, library_log = fake_worker(tmp_path, "library")
    from_environment, environment_log = fake_worker(tmp_path, "environment")
    monkeypatch.setenv("RGPOT_DXTB_PYTHON", str(from_environment))

    engine = Engine(
        os.environ["RGPOT_ENGINE_DXTB"],
        "dxtb",
        "gfn1",
        library=str(from_library),
    )
    engine.close()
    assert len(configure_events(library_log)) == 1
    assert configure_events(environment_log) == []

    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn1")
    engine.close()
    assert len(configure_events(library_log)) == 1
    assert len(configure_events(environment_log)) == 1


def test_missing_library_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.setenv(
        "RGPOT_DXTB_PYTHON", os.environ.get("RGPOT_DXTB_PYTHON") or sys.executable
    )
    with pytest.raises(RuntimeError, match="dxtb worker exited"):
        Engine(
            os.environ["RGPOT_ENGINE_DXTB"],
            "dxtb",
            "gfn1",
            library="/nonexistent/rgpot-dxtb-python",
        )


@pytest.mark.parametrize(
    "spelled,method",
    [
        ("gfn1-xtb", "gfn1"),
        ("GFN1-xTB", "gfn1"),
        ("gfn2-xtb", "gfn2"),
        ("gfn2", "gfn2"),
    ],
)
def test_method_spellings_reach_the_worker(spelled, method):
    numbers = [8, 1, 1]
    positions = np.asarray(
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
    )
    ref_energy, ref_forces = direct_force(numbers, positions, method, charge=0.0, uhf=0)
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", spelled)
    try:
        energy, forces = engine.force(numbers, positions)
        np.testing.assert_allclose(energy, ref_energy, rtol=2e-13, atol=2e-12)
        np.testing.assert_allclose(forces, ref_forces, rtol=2e-12, atol=2e-11)
    finally:
        engine.close()


def test_unknown_method_is_refused_at_creation():
    with pytest.raises(RuntimeError, match="gfn1 or gfn2"):
        Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn0")


def socket_descriptors(pid_path="/proc/self/fd"):
    found = []
    for entry in sorted(os.listdir(pid_path), key=int):
        try:
            target = os.readlink(os.path.join(pid_path, entry))
        except OSError:
            continue
        if target.startswith("socket:"):
            found.append(entry)
    return found


def test_worker_sockets_do_not_leak_into_other_children():
    engine = Engine(os.environ["RGPOT_ENGINE_DXTB"], "dxtb", "gfn1")
    try:
        assert socket_descriptors(), "the engine holds one end of its worker socket"
        listing = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; print(*[os.readlink('/proc/self/fd/' + e) for e in "
                "os.listdir('/proc/self/fd') if os.path.exists('/proc/self/fd/' + e)])",
            ],
            close_fds=False,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "socket:" not in listing.stdout
    finally:
        engine.close()


STDIO_CLOSED = r'''
import ctypes
import os
import sys

import numpy as np

engine_path, params_path, result_path = sys.argv[1:4]
lib = ctypes.CDLL(engine_path)
lib.rgpot_engine_create.restype = ctypes.c_void_p
lib.rgpot_engine_create.argtypes = [
    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t,
]
lib.rgpot_engine_destroy.argtypes = [ctypes.c_void_p]
lib.rgpot_engine_force.restype = ctypes.c_int
lib.rgpot_engine_force.argtypes = [
    ctypes.c_void_p, ctypes.c_long,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p, ctypes.c_void_p,
]
params = open(params_path, "rb").read()
storage = ctypes.create_string_buffer(params)
error = ctypes.create_string_buffer(1024)
z = np.asarray([1, 1], dtype=np.int32)
x = np.asarray([0.0, 0.0, -0.37, 0.0, 0.0, 0.37], dtype=np.float64)
forces = np.empty_like(x)
energy = ctypes.c_double()
box = np.diag([40.0, 40.0, 40.0]).astype(np.float64)
result = open(result_path, "w")

os.close(0)
os.close(1)
handle = lib.rgpot_engine_create(storage, len(params), error, len(error))
held = os.readlink("/proc/self/fd/0") if handle else ""
status = -1
if handle:
    status = lib.rgpot_engine_force(
        handle, 2,
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        forces.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(energy), None,
        box.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), None, None,
    )
    lib.rgpot_engine_destroy(handle)
result.write(repr((bool(handle), error.value.decode(), held, status, energy.value)))
result.close()
os._exit(0)
'''


def test_worker_starts_when_stdin_and_stdout_are_closed(tmp_path):
    message = SCHEMA.BuiltinParams.new_message()
    message.name = "dxtb"
    xtb = message.init("config").init("xtb")
    xtb.method = "gfn1"
    xtb.accuracy = 1.0
    xtb.electronicTemperature = 300.0
    xtb.maxIterations = 300
    params_path = tmp_path / "params.bin"
    params_path.write_bytes(message.to_bytes())
    script_path = tmp_path / "stdio_closed.py"
    script_path.write_text(STDIO_CLOSED)
    result_path = tmp_path / "result.txt"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            os.environ["RGPOT_ENGINE_DXTB"],
            str(params_path),
            str(result_path),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    created, error, held, status, energy = ast.literal_eval(result_path.read_text())
    assert created, error
    # Descriptors 0 and 1 were free, so the engine's own socket took them.
    assert held.startswith("socket:")
    assert status == 0
    ref_energy, _ = direct_force(
        [1, 1], [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]], "gfn1", charge=0.0, uhf=0
    )
    np.testing.assert_allclose(energy, ref_energy, rtol=2e-13, atol=2e-12)


def _events(log_path):
    if not log_path.exists():
        return []
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_shutdown_treats_a_closed_peer_as_clean_and_a_live_drop_as_an_error(tmp_path):
    """Repeat shutdown while the worker closes, truncates, or writes late.

    A broken pipe during that shutdown must not fail the host. The same
    condition on a live force call must still be reported.
    """
    cycles = 20
    for mode in ("late-write", "drop", "partial"):
        for i in range(cycles):
            wrapper, log_path = fake_worker(tmp_path, f"{mode}-{i}", mode=mode)
            engine = Engine(
                os.environ["RGPOT_ENGINE_DXTB"],
                "dxtb",
                "gfn1",
                library=str(wrapper),
            )
            engine.close()
            events = _events(log_path)
            assert any(ev.get("event") == "shutdown" for ev in events), events
            assert not any(ev.get("result") == "broken-pipe" for ev in events), events

    positions = np.asarray(
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]
    )
    for i in range(cycles):
        wrapper, log_path = fake_worker(tmp_path, f"live-{i}", mode="drop-force")
        engine = Engine(
            os.environ["RGPOT_ENGINE_DXTB"],
            "dxtb",
            "gfn1",
            library=str(wrapper),
        )
        try:
            status = engine.force_status([8, 1, 1], positions)
            assert status != 0
            assert "exited" in engine.last_error()
        finally:
            engine.close()
        assert not any(ev.get("result") == "broken-pipe" for ev in _events(log_path))


def _reaped(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.parametrize("mode", ["hang-shutdown", "linger"])
def test_shutdown_kills_a_worker_that_does_not_exit(tmp_path, mode):
    """A worker that never acknowledges, or never exits, cannot hang destroy."""
    wrapper, log_path = fake_worker(tmp_path, mode, mode=mode)
    engine = Engine(
        os.environ["RGPOT_ENGINE_DXTB"],
        "dxtb",
        "gfn1",
        library=str(wrapper),
    )
    (pid,) = [ev["pid"] for ev in configure_events(log_path)]
    started = time.monotonic()
    engine.close()
    elapsed = time.monotonic() - started
    assert elapsed < 30.0, elapsed
    assert any(ev.get("event") == "shutdown" for ev in _events(log_path))
    assert _reaped(pid)
