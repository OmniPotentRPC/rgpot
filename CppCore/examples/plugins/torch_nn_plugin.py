#!/usr/bin/env python3
"""
rgpot plugin: neural network potential via PyTorch.

A minimal feed-forward neural network that maps pairwise distances to
per-atom energies. Forces are computed via torch.autograd. This is NOT
a production potential -- it demonstrates the pattern for wrapping any
PyTorch model as an rgpot plugin.

The network architecture:
    input:  pairwise distance features [n_atoms, n_features]
    hidden: 2 layers of 32 units, SiLU activation
    output: per-atom energy [n_atoms, 1]
    total:  sum of per-atom energies
    forces: -grad(total_energy, positions)

Requirements: pip install torch numpy

Usage:
    from torch_nn_plugin import TorchNNPotential
    pot = TorchNNPotential()
    energy, forces = pot.calculate(positions, atomic_numbers, cell)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class PairwiseNN(nn.Module):
    """Simple pairwise distance NN for demonstration."""

    def __init__(self, n_features: int = 10, cutoff: float = 6.0):
        super().__init__()
        self.cutoff = cutoff
        self.n_features = n_features

        # Gaussian basis expansion parameters
        centers = torch.linspace(0.5, cutoff - 0.5, n_features)
        self.register_buffer("centers", centers)
        self.width = 0.5

        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total energy from positions [n_atoms, 3]."""
        n = positions.shape[0]

        # Pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [n, n, 3]
        dist = torch.norm(diff, dim=-1)  # [n, n]

        # Mask self-interactions and beyond cutoff
        mask = (dist > 1e-8) & (dist < self.cutoff)

        # Gaussian basis expansion of distances
        rbf = torch.exp(
            -((dist.unsqueeze(-1) - self.centers) ** 2) / (2 * self.width**2)
        )  # [n, n, n_features]

        # Zero out masked entries
        rbf = rbf * mask.unsqueeze(-1).float()

        # Sum over neighbors to get per-atom features
        atom_features = rbf.sum(dim=1)  # [n, n_features]

        # Per-atom energies
        atom_energies = self.net(atom_features)  # [n, 1]

        return atom_energies.sum()


class TorchNNPotential:
    """PyTorch NN potential with autograd forces."""

    def __init__(
        self,
        model_path: str | None = None,
        cutoff: float = 6.0,
        device: str = "cpu",
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required: pip install torch")

        self.device = torch.device(device)

        if model_path:
            self.model = torch.jit.load(model_path, map_location=self.device)
        else:
            self.model = PairwiseNN(cutoff=cutoff).to(self.device)
            self.model.eval()

    def calculate(
        self,
        positions: np.ndarray,
        atomic_numbers: np.ndarray,
        cell: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return (energy, forces) using autograd."""
        pos_t = torch.tensor(
            positions, dtype=torch.float64, device=self.device, requires_grad=True
        )

        # Forward pass
        energy = self.model(pos_t)

        # Backward pass for forces
        energy.backward()
        forces = -pos_t.grad.detach().cpu().numpy()
        energy_val = energy.item()

        return energy_val, forces.ravel()


# -- ctypes bridge --
_instances: dict[int, TorchNNPotential] = {}
_next_id = 1


def plugin_create(config_json: str) -> int:
    import json

    global _next_id
    cfg = json.loads(config_json) if config_json else {}
    pot = TorchNNPotential(
        model_path=cfg.get("model_path"),
        cutoff=cfg.get("cutoff", 6.0),
        device=cfg.get("device", "cpu"),
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
    torch.manual_seed(42)

    pos = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=np.float64
    )
    atm = np.array([1, 1, 1])
    cell = np.eye(3) * 10.0

    pot = TorchNNPotential(cutoff=6.0)
    e, f = pot.calculate(pos, atm, cell)
    print(f"Energy: {e:.10f}")
    print(f"Forces:\n{f.reshape(-1, 3)}")

    # Verify forces via finite differences
    h = 1e-5
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
    assert max_err < 1e-4, f"Force error too large: {max_err}"
    print("Finite difference check: OK")

    # Verify forces sum to ~zero
    fsum = np.abs(f.reshape(-1, 3).sum(axis=0))
    print(f"Force sum: {fsum}")
    print("All checks passed.")
