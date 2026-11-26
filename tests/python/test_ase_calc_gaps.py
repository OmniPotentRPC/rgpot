# Copyright 2023--present rgpot developers

"""ASE 3.29 calculator contract points that one frame of stress did not cover."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.db import connect

import rgpot
from rgpot.ase import RgpotCalculator

ATOL = 1e-9
LJ = {"u0": 0.0104, "psi": 3.40, "cutoff": 8.5}


class _CallCount:
    """Counts kernel calls without changing the cutoff or the capabilities."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.evaluate_calls = 0
        self.per_atom_calls = 0
        # The calculator looks up a cutoff by class name for kernels such as
        # EMT. A wrapper has to publish the same number itself.
        from rgpot.ase.calculator import _KERNEL_CUTOFF_ANGSTROM

        cfg = inner.config() if callable(getattr(inner, "config", None)) else {}
        if isinstance(cfg, dict) and cfg.get("cutoff") is not None:
            self.cutoff = float(cfg["cutoff"])
        else:
            self.cutoff = float(_KERNEL_CUTOFF_ANGSTROM[type(inner).__name__])
        if not hasattr(inner, "evaluate_per_atom"):
            self.evaluate_per_atom = None

    def caps(self):
        return self._inner.caps()

    def config(self):
        return self._inner.config()

    def evaluate(self, positions, numbers, box):
        self.evaluate_calls += 1
        return self._inner.evaluate(positions, numbers, box)

    def evaluate_per_atom(self, positions, numbers, box):
        self.per_atom_calls += 1
        return self._inner.evaluate_per_atom(positions, numbers, box)


def _argon():
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True)
    atoms.rattle(0.04, seed=3)
    return atoms


def test_numerical_stress_parameters_round_trip_through_set_and_todict() -> None:
    """stress_step and numerical_stress belong to the calculator, not the kernel."""
    calc = RgpotCalculator("emt", numerical_stress=True, stress_step=2.5e-6)
    assert calc.parameters.numerical_stress is True
    assert calc.parameters.stress_step == pytest.approx(2.5e-6)
    data = calc.todict()
    assert data["numerical_stress"] is True
    assert data["stress_step"] == pytest.approx(2.5e-6)
    rebuilt = RgpotCalculator(**data)
    assert rebuilt.parameters.numerical_stress is True
    assert rebuilt.parameters.stress_step == pytest.approx(2.5e-6)
    assert "stress" in rebuilt.implemented_properties
    assert rebuilt.todict()["numerical_stress"] is True

    plain = RgpotCalculator("emt")
    assert plain.parameters.numerical_stress is False
    assert plain.parameters.stress_step == pytest.approx(1e-6)
    assert "numerical_stress" not in plain.todict()
    assert "stress_step" not in plain.todict()
    again = RgpotCalculator(**plain.todict())
    assert again.parameters.numerical_stress is False
    assert again.parameters.stress_step == pytest.approx(1e-6)

    # set must not forward these names into the EMT constructor.
    plain.set(numerical_stress=True, stress_step=3e-6)
    assert plain.parameters.numerical_stress is True
    assert plain.parameters.stress_step == pytest.approx(3e-6)
    assert "stress" in plain.implemented_properties
    plain.set(numerical_stress=False)
    assert "stress" not in plain.implemented_properties


