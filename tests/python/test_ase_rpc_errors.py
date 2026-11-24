# MIT License
# Copyright 2023--present rgpot developers
"""ASE error contract of RgpotRpcCalculator.calculate, without a server."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from ase.build import bulk
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    InputError,
    PropertyNotImplementedError,
)

pytest.importorskip("capnp")

from rgpot.ase import RgpotRpcCalculator
from rgpot.ase.claim import ConnectionClaim
from rgpot.ase.rpc import _capnp, _schema_path


class _FakePotential:
    """Stand-in for the capnp Potential capability."""

    def __init__(self, stress: list[float], fail: Exception | None = None) -> None:
        self._stress = stress
        self._fail = fail

    async def _reply(self, natoms: int) -> Any:
        if self._fail is not None:
            raise self._fail
        result = SimpleNamespace(
            energy=-1.0,
            forces=[0.0] * (3 * natoms),
            stress=self._stress,
        )
        return SimpleNamespace(result=result)

    async def calculate(self, fip: Any) -> Any:
        return await self._reply(len(fip.atmnrs))

    async def calculateStress(self, fip: Any) -> Any:
        return await self._reply(len(fip.atmnrs))


class _OfflineCalculator(RgpotRpcCalculator):
    """Calculator wired to a fake capability instead of a socket."""

    def __init__(self, pot: _FakePotential) -> None:
        Calculator.__init__(self)
        self._claim = ConnectionClaim("RgpotRpcCalculator")
        self._schema = _capnp().load(str(_schema_path()))
        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._pot = pot
        self._server_proc = None
        self._orbital_key = None
        self._server_has_stress = True

    async def _refresh_capabilities_async(self) -> None:
        self._server_has_stress = True
        self.implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._loop.close()


def _atoms() -> Any:
    return bulk("Ar", "fcc", a=5.26, cubic=True)


def test_short_stress_reply_raises_property_not_implemented() -> None:
    calc = _OfflineCalculator(_FakePotential(stress=[0.0] * 6))
    try:
        atoms = _atoms()
        atoms.calc = calc
        with pytest.raises(PropertyNotImplementedError, match="returned no stress"):
            atoms.get_stress()
        with pytest.raises(PropertyNotImplementedError):
            calc.calculate(atoms, ["stress"], ["positions"])
    finally:
        calc.close()


def test_empty_stress_reply_still_serves_energy_and_forces() -> None:
    calc = _OfflineCalculator(_FakePotential(stress=[]))
    try:
        atoms = _atoms()
        atoms.calc = calc
        assert atoms.get_potential_energy() == -1.0
        assert atoms.get_forces().shape == (len(atoms), 3)
    finally:
        calc.close()


def test_calculator_errors_pass_through_unchanged() -> None:
    calc = _OfflineCalculator(_FakePotential(stress=[]))
    failure = InputError("bad input")

    async def _raise(*args: Any, **kwargs: Any) -> Any:
        raise failure

    calc._calculate_async = _raise  # type: ignore[method-assign]
    try:
        atoms = _atoms()
        atoms.calc = calc
        with pytest.raises(CalculatorError) as caught:
            atoms.get_potential_energy()
        assert caught.value is failure
    finally:
        calc.close()


def test_transport_failure_is_a_runtime_error() -> None:
    cause = ConnectionError("peer closed")
    calc = _OfflineCalculator(_FakePotential(stress=[], fail=cause))
    try:
        atoms = _atoms()
        atoms.calc = calc
        with pytest.raises(RuntimeError, match="RPC calculate failed") as caught:
            atoms.get_potential_energy()
        assert not isinstance(caught.value, (CalculatorError, NotImplementedError))
        root = caught.value
        while root.__cause__ is not None:
            root = root.__cause__
        assert root is cause
    finally:
        calc.close()
