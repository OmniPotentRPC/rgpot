#!/usr/bin/env python3
"""
rgpot plugin: harmonic spring potential (pure Python via ctypes).

Each atom is tethered to its initial position by a harmonic spring.
Energy = 0.5 * k * sum(|r - r0|^2), Forces = -k * (r - r0).

Build: not needed -- Python is interpreted.
Use:   Set RGPOT_PLUGIN_PATH to the directory containing the compiled
       wrapper (see harmonic_ctypes_wrapper.c), or load this module
       directly from Python.

This file demonstrates how to implement an rgpot plugin entirely in
Python using ctypes. The actual .so that rgpot discovers is a tiny
C wrapper (harmonic_ctypes_wrapper.c) that embeds Python and calls
back into this module.

For simpler use, this module can also be used standalone:

    from harmonic_plugin import HarmonicPotential
    pot = HarmonicPotential(k=1.0)
    energy, forces = pot.calculate(positions, atomic_numbers, cell)
"""

import numpy as np


class HarmonicPotential:
    """Harmonic tether potential: E = 0.5*k*sum(|r-r0|^2)."""

    def __init__(self, k: float = 1.0):
        self.k = k
        self.r0: np.ndarray | None = None

    def calculate(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return (energy, forces) for the given configuration."""
        if self.r0 is None:
            self.r0 = positions.copy()

        disp = positions - self.r0
        energy = 0.5 * self.k * np.sum(disp**2)
        forces = -self.k * disp
        return float(energy), forces


# -- ctypes bridge for use as an rgpot plugin .so wrapper --
# The C wrapper calls these functions via Python/C API.

_instances: dict[int, HarmonicPotential] = {}
_next_id = 1


def plugin_create(config_json: str) -> int:
    """Create an instance, return integer handle."""
    import json

    global _next_id
    cfg = json.loads(config_json) if config_json else {}
    pot = HarmonicPotential(k=cfg.get("k", 1.0))
    handle = _next_id
    _next_id += 1
    _instances[handle] = pot
    return handle


def plugin_destroy(handle: int) -> None:
    """Destroy an instance by handle."""
    _instances.pop(handle, None)


def plugin_calculate(
    handle: int,
    n_atoms: int,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calculate energy and forces."""
    pot = _instances[handle]
    pos = positions.reshape(n_atoms, 3)
    energy, forces = pot.calculate(pos, atomic_numbers, cell.reshape(3, 3))
    return energy, forces.ravel()


if __name__ == "__main__":
    # Quick self-test
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    atm = np.array([1, 1, 1])
    cell = np.eye(3) * 10.0

    pot = HarmonicPotential(k=2.0)
    e, f = pot.calculate(pos, atm, cell)
    print(f"Initial: E={e:.6f} (expect 0.0)")
    assert abs(e) < 1e-12

    pos2 = pos + 0.1
    e2, f2 = pot.calculate(pos2, atm, cell)
    print(f"Displaced: E={e2:.6f} (expect 0.09)")
    print(f"Forces:\n{f2}")
    assert abs(e2 - 0.09) < 1e-12
    print("All checks passed.")
