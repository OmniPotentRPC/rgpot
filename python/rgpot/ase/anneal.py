"""Portfolio relaxation helper.

:class:`AnnealRunner` is not an :class:`ase.optimize.Optimizer`. It calls
``anneal.global_optimize`` and writes the best geometry back onto the
atoms. :class:`AnnealOptimizer` is the deprecated name.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from ase import Atoms


class AnnealRunner:
    """Relax an :class:`ase.Atoms` with anneal's global portfolio optimizer.

    This is not an :class:`ase.optimize.Optimizer`. It does not implement
    the ASE optimizer step protocol. Construct it with atoms that already
    have a calculator, call :meth:`run`, and the atoms are updated in place.

    Parameters
    ----------
    atoms:
        Structure to optimize; must have a calculator attached.
    budget:
        Combined objective and gradient evaluation budget.
    seed:
        RNG seed.
    pad:
        Search-box half-width (Angstrom) around the starting coordinates.
    use_forces:
        Pass the calculator forces in as the analytic gradient.
    """

    def __init__(
        self,
        atoms: Atoms,
        *,
        budget: int = 20000,
        seed: int = 0,
        pad: float = 3.0,
        use_forces: bool = True,
    ) -> None:
        if atoms.calc is None:
            raise ValueError("AnnealRunner needs atoms with a calculator attached")
        self.atoms = atoms
        self.budget = int(budget)
        self.seed = int(seed)
        self.pad = float(pad)
        self.use_forces = bool(use_forces)
        self.result: dict[str, Any] | None = None

    def _energy(self, x: np.ndarray) -> float:
        self.atoms.set_positions(np.asarray(x).reshape(-1, 3))
        return float(self.atoms.get_potential_energy())

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        self.atoms.set_positions(np.asarray(x).reshape(-1, 3))
        return (-self.atoms.get_forces()).ravel()

    def run(self) -> Atoms:
        """Optimize and write the best geometry back into ``atoms``."""
        import anneal

        x0 = self.atoms.get_positions().ravel()
        low = x0 - self.pad
        high = x0 + self.pad
        grad = self._gradient if self.use_forces else None
        self.result = anneal.global_optimize(
            self._energy, low, high, self.budget, seed=self.seed, grad_fn=grad
        )
        self.atoms.set_positions(np.asarray(self.result["best_pos"]).reshape(-1, 3))
        return self.atoms

    def get_potential_energy(self) -> float:
        """Best energy found, after :meth:`run`."""
        if self.result is None:
            raise RuntimeError("call run() first")
        return float(self.result["best_val"])

    @property
    def nevals(self) -> int:
        if self.result is None:
            return 0
        return int(self.result["n_evals"]) + int(self.result["n_grads"])


class AnnealOptimizer(AnnealRunner):
    """Deprecated name of :class:`AnnealRunner`.

    Not an :class:`ase.optimize.Optimizer`.
    """

    def __init__(
        self,
        atoms: Atoms,
        *,
        budget: int = 20000,
        seed: int = 0,
        pad: float = 3.0,
        use_forces: bool = True,
    ) -> None:
        warnings.warn(
            "AnnealOptimizer is deprecated; use AnnealRunner. "
            "It is not an ase.optimize.Optimizer.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            atoms, budget=budget, seed=seed, pad=pad, use_forces=use_forces
        )
