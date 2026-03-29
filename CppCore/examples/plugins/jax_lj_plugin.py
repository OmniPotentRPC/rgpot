#!/usr/bin/env python3
"""
rgpot plugin: Lennard-Jones potential with automatic differentiation via JAX.

This demonstrates implementing an rgpot plugin in Python where forces are
computed via JAX's reverse-mode autodiff (jax.grad) rather than hand-coded
analytic derivatives.

Energy: sum_{i<j} 4*epsilon*((sigma/r_ij)^12 - (sigma/r_ij)^6)
Forces: -grad(Energy) w.r.t. positions (computed by JAX)

Requirements: pip install jax jaxlib numpy

Usage as standalone Python:
    from jax_lj_plugin import JaxLJPotential
    pot = JaxLJPotential(epsilon=1.0, sigma=1.0, cutoff=15.0)
    energy, forces = pot.calculate(positions, atomic_numbers, cell)
"""

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    # Use float64 for scientific accuracy
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


class JaxLJPotential:
    """Lennard-Jones potential with JAX autodiff forces."""

    def __init__(
        self,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        cutoff: float = 15.0,
    ):
        if not HAS_JAX:
            raise ImportError("JAX is required: pip install jax jaxlib")
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff

        # JIT-compile the energy function and its gradient
        self._energy_fn = jax.jit(self._energy)
        self._grad_fn = jax.jit(jax.grad(self._energy))

    def _energy(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Compute LJ energy (differentiable)."""
        n = positions.shape[0]
        e = jnp.float64(0.0)

        # Pairwise distances (upper triangle)
        for i in range(n):
            for j in range(i + 1, n):
                dr = positions[i] - positions[j]
                r = jnp.sqrt(jnp.sum(dr**2))
                r = jnp.maximum(r, 1e-10)  # avoid division by zero
                inv_r6 = (self.sigma / r) ** 6
                e = e + jnp.where(
                    r < self.cutoff,
                    4.0 * self.epsilon * inv_r6 * (inv_r6 - 1.0),
                    0.0,
                )
        return e

    def calculate(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return (energy, forces)."""
        pos_jax = jnp.array(positions)
        energy = float(self._energy_fn(pos_jax))
        grad = np.asarray(self._grad_fn(pos_jax))
        forces = -grad  # F = -dE/dr
        return energy, forces


# -- ctypes bridge --
_instances: dict[int, JaxLJPotential] = {}
_next_id = 1


def plugin_create(config_json: str) -> int:
    import json

    global _next_id
    cfg = json.loads(config_json) if config_json else {}
    pot = JaxLJPotential(
        epsilon=cfg.get("epsilon", 1.0),
        sigma=cfg.get("sigma", 1.0),
        cutoff=cfg.get("cutoff", 15.0),
    )
    handle = _next_id
    _next_id += 1
    _instances[handle] = pot
    return handle


def plugin_destroy(handle: int) -> None:
    _instances.pop(handle, None)


def plugin_calculate(
    handle: int,
    n_atoms: int,
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, np.ndarray]:
    pot = _instances[handle]
    pos = positions.reshape(n_atoms, 3)
    energy, forces = pot.calculate(pos, atomic_numbers, cell.reshape(3, 3))
    return energy, forces.ravel()


if __name__ == "__main__":
    pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    atm = np.array([1, 1])
    cell = np.eye(3) * 20.0

    pot = JaxLJPotential(epsilon=1.0, sigma=1.0, cutoff=15.0)
    e, f = pot.calculate(pos, atm, cell)
    print(f"Energy: {e:.10f}")
    print(f"Forces:\n{f.reshape(-1, 3)}")
    # Forces should be equal and opposite (Newton's third law)
    assert np.allclose(f[:3] + f[3:], 0.0, atol=1e-10)
    print("Newton's third law: OK")

    # Verify forces via finite differences
    h = 1e-6
    fd_forces = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_p = pos.copy()
            pos_m = pos.copy()
            pos_p[i, j] += h
            pos_m[i, j] -= h
            ep, _ = pot.calculate(pos_p, atm, cell)
            em, _ = pot.calculate(pos_m, atm, cell)
            fd_forces[i, j] = -(ep - em) / (2 * h)

    max_err = np.max(np.abs(f.reshape(-1, 3) - fd_forces))
    print(f"Max force error vs finite diff: {max_err:.2e}")
    assert max_err < 1e-5, f"Force error too large: {max_err}"
    print("Finite difference check: OK")
