# Copyright 2023--present rgpot developers

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import re
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


class ForceInputMessage:
    hasOrbitalKey = False
    orbitalKey = 0


class ForceInputFactory:
    @staticmethod
    def new_message() -> ForceInputMessage:
        return ForceInputMessage()


class Schema:
    ForceInput = ForceInputFactory


SCHEMA = Schema()


def _rpc_calculator_class(current_minor: int | None = None):
    rgpot_package = types.ModuleType("rgpot")
    rgpot_package.__path__ = [str(ROOT / "python/rgpot")]
    ase_package = types.ModuleType("rgpot.ase")
    ase_package.__path__ = [str(ROOT / "python/rgpot/ase")]
    sys.modules.setdefault("rgpot", rgpot_package)
    sys.modules.setdefault("rgpot.ase", ase_package)
    for name in ("claim", "rpc"):
        qualified = f"rgpot.ase.{name}"
        spec = importlib.util.spec_from_file_location(
            qualified, ROOT / f"python/rgpot/ase/{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        if name == "rpc" and current_minor is not None:
            source, count = re.subn(
                r"^PROTOCOL_MINOR = \d+$",
                f"PROTOCOL_MINOR = {current_minor}",
                Path(spec.origin).read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            assert count == 1
            exec(compile(source, spec.origin, "exec"), module.__dict__)
        else:
            spec.loader.exec_module(module)
    return sys.modules["rgpot.ase.rpc"].RgpotRpcCalculator


class KeyedPotential:
    def __init__(self, protocol_minor: int = 2) -> None:
        self.generations: dict[int, int] = {}
        self.keys: list[int] = []
        self.stress_keys: list[int] = []
        self.protocol_minor = protocol_minor
        self.capabilities_calls = 0

    async def getCapabilities(self):
        self.capabilities_calls += 1

        class Capabilities:
            protocolFamily = "rgpot.potentials"
            protocolMajor = 1
            protocolMinor = self.protocol_minor

        class Reply:
            capabilities = Capabilities()

        return Reply()

    async def calculate(self, fip):
        assert fip.hasOrbitalKey
        key = int(fip.orbitalKey)
        self.keys.append(key)
        generation = self.generations.get(key, 0) + 1
        self.generations[key] = generation

        class PotentialResult:
            energy = float(generation)
            forces = [0.0, 0.0, 0.0]
            stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        class Reply:
            result = PotentialResult()

        return Reply()

    async def calculateStress(self, fip):
        self.stress_keys.append(int(fip.orbitalKey))
        reply = await self.calculate(fip)
        reply.result.stress = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        return reply


class BaselinePotential(KeyedPotential):
    async def calculate(self, fip):
        assert not fip.hasOrbitalKey

        class PotentialResult:
            energy = 1.0
            forces = [0.0, 0.0, 0.0]

        class Reply:
            result = PotentialResult()

        return Reply()


def _calculator(potential, current_minor: int | None = None):
    RgpotRpcCalculator = _rpc_calculator_class(current_minor)
    calculator = object.__new__(RgpotRpcCalculator)
    calculator._schema = SCHEMA
    calculator._pot = potential
    calculator._server_protocol_minor = None
    return calculator


def test_minor_two_client_uses_orbital_key_on_minor_two_server() -> None:
    calculator = _calculator(KeyedPotential(protocol_minor=2))

    async def run() -> list[float]:
        energies = []
        for key in (7, 19, 7):
            evaluated = await calculator._calculate_async(
                np.zeros((1, 3)), [8], np.eye(3), key
            )
            energies.append(evaluated["energy"])
        return energies

    assert asyncio.run(run()) == [1.0, 1.0, 2.0]
    assert calculator._pot.keys == [7, 19, 7]
    assert calculator._pot.capabilities_calls == 1


def test_python_rpc_stress_request_keeps_the_orbital_key() -> None:
    RgpotRpcCalculator = _rpc_calculator_class()
    calculator = object.__new__(RgpotRpcCalculator)
    calculator._schema = SCHEMA
    calculator._pot = KeyedPotential()

    async def run() -> dict:
        return await calculator._calculate_async(
            np.zeros((1, 3)),
            [8],
            np.eye(3),
            7,
            request_stress=True,
            accept_stress=True,
        )

    evaluated = asyncio.run(run())
    assert evaluated["energy"] == 1.0
    assert calculator._pot.stress_keys == [7]
    assert calculator._pot.keys == [7]
    assert list(evaluated["stress"]) == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]


def test_minor_two_client_rejects_orbital_key_on_minor_one_server() -> None:
    calculator = _calculator(KeyedPotential(protocol_minor=1))

    with pytest.raises(
        RuntimeError,
        match=r"ForceInput orbital key requires RPC protocol minor 2.*advertises 1",
    ):
        asyncio.run(
            calculator._calculate_async(np.zeros((1, 3)), [8], np.eye(3), 7)
        )
    assert calculator._pot.keys == []
    assert calculator._pot.capabilities_calls == 1


def test_minor_two_client_uses_baseline_force_input_on_minor_one_server() -> None:
    calculator = _calculator(BaselinePotential(protocol_minor=1))

    evaluated = asyncio.run(
        calculator._calculate_async(np.zeros((1, 3)), [8], np.eye(3))
    )

    assert evaluated["energy"] == 1.0
    assert evaluated["forces"].shape == (1, 3)
    assert calculator._pot.capabilities_calls == 0


@pytest.mark.parametrize("current_minor", [2, 3, 7])
def test_orbital_key_gate_does_not_follow_the_current_minor(
    current_minor: int,
) -> None:
    accepted = _calculator(
        KeyedPotential(protocol_minor=2), current_minor=current_minor
    )
    rpc = sys.modules["rgpot.ase.rpc"]
    assert rpc.PROTOCOL_MINOR == current_minor
    assert rpc.ORBITAL_KEY_INTRODUCED_MINOR == 2

    evaluated = asyncio.run(
        accepted._calculate_async(np.zeros((1, 3)), [8], np.eye(3), 7)
    )
    assert evaluated["energy"] == 1.0
    assert accepted._pot.keys == [7]

    refused = _calculator(
        KeyedPotential(protocol_minor=1), current_minor=current_minor
    )
    with pytest.raises(
        RuntimeError,
        match=r"ForceInput orbital key requires RPC protocol minor 2.*advertises 1",
    ):
        asyncio.run(
            refused._calculate_async(np.zeros((1, 3)), [8], np.eye(3), 7)
        )
    assert refused._pot.keys == []
