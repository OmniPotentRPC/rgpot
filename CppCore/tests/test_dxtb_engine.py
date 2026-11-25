# Copyright 2023--present rgpot developers
"""Persistent dxtb engine checks against dxtb 0.4.0 autograd."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
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
