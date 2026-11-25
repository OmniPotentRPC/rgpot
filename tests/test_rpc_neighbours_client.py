# Copyright 2023--present rgpot developers

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys
import types

from ase import Atoms
import pytest

ROOT = Path(__file__).parents[1]


class ForceInputMessage:
    pass


class NeighborRequestMessage:
    cutoff = 0.0
    skin = 0.0
    format = 0
    cut = 0
    periodic: list[bool] = []
    shifts = False
    vectors = False
    distances = False
    mode = 0


class Schema:
    class ForceInput:
        @staticmethod
        def new_message() -> ForceInputMessage:
            return ForceInputMessage()

    class NeighborRequest:
        @staticmethod
        def new_message() -> NeighborRequestMessage:
            return NeighborRequestMessage()


_LOADED = None


def _install_source_tree() -> None:
    """Make ``rgpot._schema`` importable from this checkout.

    The repository root is on ``sys.path`` during pytest, so ``import rgpot``
    binds the ``rgpot/`` directory as a namespace package. That package has
    no ``_schema``. ``setdefault`` would keep it, and loading ``rpc.py``
    would stop before ``RgpotRpcCalculator`` exists. A regular installed
    package (one with ``__file__``) is left in place.
    """
    package = sys.modules.get("rgpot")
    if package is not None and getattr(package, "__file__", None):
        return
    source = types.ModuleType("rgpot")
    source.__path__ = [str(ROOT / "python/rgpot")]
    source.__package__ = "rgpot"
    sys.modules["rgpot"] = source
    sys.modules.pop("rgpot._schema", None)
    ase = types.ModuleType("rgpot.ase")
    ase.__path__ = [str(ROOT / "python/rgpot/ase")]
    ase.__package__ = "rgpot.ase"
    sys.modules["rgpot.ase"] = ase


def _rpc_module():
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    _install_source_tree()
    for name in ("claim", "rpc"):
        qualified = f"rgpot.ase.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified, ROOT / f"python/rgpot/ase/{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    _LOADED = sys.modules["rgpot.ase.rpc"]
    return _LOADED


class NeighbourPotential:
    def __init__(
        self,
        protocol_minor: int = 3,
        operations: list[str] | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self.protocol_minor = protocol_minor
        self.operations = ["energy", "forces"] if operations is None else operations
        self.fail = fail
        self.neighbour_calls = 0
        self.cutoffs: list[float] = []

    async def getCapabilities(self):
        potential = self

        class Reply:
            class capabilities:
                protocolFamily = "rgpot.potentials"
                protocolMajor = 1
                protocolMinor = potential.protocol_minor
                operations = [
                    types.SimpleNamespace(name=name) for name in potential.operations
                ]

        return Reply()

    async def neighbours(self, _fip, req):
        self.neighbour_calls += 1
        self.cutoffs.append(float(req.cutoff))
        if self.fail is not None:
            raise self.fail

        class Neighbors:
            status = 0
            message = "ok"
            first = [0]
            second = [1]
            shift = [0, 0, 0]
            vector = [0.0, 0.0, 1.0]
            distance = [1.0]

        class Reply:
            neighbors = Neighbors()

        return Reply()


def _calculator(potential: NeighbourPotential):
    rpc = _rpc_module()
    calculator = object.__new__(rpc.RgpotRpcCalculator)
    calculator._schema = Schema
    calculator._pot = potential
    calculator._server_protocol_minor = None
    calculator._server_has_neighbors = False
    calculator._claim = sys.modules["rgpot.ase.claim"].ConnectionClaim(
        "RgpotRpcCalculator"
    )
    calculator._loop = asyncio.new_event_loop()
    return calculator


def test_neighbours_refuses_a_minor_2_server_before_the_call() -> None:
    rpc = _rpc_module()
    potential = NeighbourPotential(protocol_minor=2, operations=["neighbors"])
    calculator = _calculator(potential)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], cell=[10, 10, 10])
    try:
        with pytest.raises(rpc.ProtocolMinorMismatchError, match="neighbours") as caught:
            calculator.neighbours(atoms, cutoff=1.5)
    finally:
        calculator._loop.close()
    assert caught.value.required_minor == 3
    assert caught.value.server_minor == 2
    assert potential.neighbour_calls == 0
    assert "RPC failed" not in str(caught.value)


def test_neighbours_requires_the_neighbors_operation() -> None:
    potential = NeighbourPotential(protocol_minor=3, operations=["energy", "forces"])
    calculator = _calculator(potential)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        with pytest.raises(RuntimeError, match="Capabilities.Operation.neighbors"):
            calculator.neighbours(atoms, cutoff=1.5)
    finally:
        calculator._loop.close()
    assert potential.neighbour_calls == 0


def test_neighbours_wraps_a_capnp_failure() -> None:
    potential = NeighbourPotential(
        protocol_minor=3,
        operations=["energy", "forces", "neighbors"],
        fail=RuntimeError("capnp UNIMPLEMENTED"),
    )
    calculator = _calculator(potential)
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    try:
        with pytest.raises(RuntimeError, match="neighbours RPC failed") as caught:
            calculator.neighbours(atoms, cutoff=1.5)
    finally:
        calculator._loop.close()
    assert caught.value.__cause__ is potential.fail
    assert potential.neighbour_calls == 1


def test_explicit_zero_cutoff_is_forwarded() -> None:
    rpc = _rpc_module()
    calculator = object.__new__(rpc.RgpotRpcCalculator)
    atoms = Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    assert calculator._resolve_cutoff(None, [True, True, True], 0.0, 0.0) == 0.0
    assert calculator._resolve_cutoff(atoms, [True, True, True], 0.0, 0.0) == 0.0
    assert calculator._resolve_cutoff(atoms, [True, True, True], 2.5, 0.0) == 2.5
    assert calculator._resolve_cutoff(atoms, [True, True, True], None, 0.0) == 5.0

    potential = NeighbourPotential(
        protocol_minor=3, operations=["energy", "forces", "neighbors"]
    )
    calculator = _calculator(potential)
    try:
        result = calculator.neighbours(atoms, cutoff=0.0, shifts=False, vectors=False)
    finally:
        calculator._loop.close()
    assert potential.cutoffs == [0.0]
    assert list(result["first"]) == [0]
    assert list(result["distance"]) == [1.0]
