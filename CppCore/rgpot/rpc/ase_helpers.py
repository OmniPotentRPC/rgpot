#!/usr/bin/env python3
"""Ergonomic ASE helpers for the rgpot + anneal + readcon stack.

These wrap the lower-level pieces so a user works in plain ASE terms:

* :class:`AnnealOptimizer` -- an ASE-optimizer-style driver. Construct it with an
  :class:`ase.Atoms` that already has a calculator, call :meth:`run`, and the
  atoms are relaxed in place by anneal's global portfolio optimizer (the INFORMS
  Journal on Computing method) using the calculator's energy and forces. The API
  mirrors ``ase.optimize`` optimizers (``opt = AnnealOptimizer(atoms); opt.run()``).

* :func:`atoms_from_con` / :func:`atoms_to_con` / :func:`atoms_to_calc` --
  load and write atomic structures as EON ``.con`` files via ``readcon``
  (readcon-core's Python module). Loading keeps the
  ``SinglePointCalculator`` that ``to_ase`` attached unless the caller passes
  ``calc``. Writing uses ``ConFrame.from_ase`` and ``write_con`` with
  precision omitted and canonical left off, and does not open a corpus.

The heavy dependencies (``anneal``, ``readcon``) are imported lazily so importing
this module only needs ASE + numpy.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


class AnnealOptimizer:
    """Relax an :class:`ase.Atoms` with anneal's global portfolio optimizer.

    Parameters
    ----------
    atoms:
        Structure to optimize; must have a calculator attached.
    budget:
        Combined objective + gradient evaluation budget for the portfolio.
    seed:
        RNG seed (deterministic runs).
    pad:
        Search-box half-width (Angstrom) around the starting coordinates; the
        portfolio explores ``x0 +/- pad`` per Cartesian component.
    use_forces:
        Pass the calculator's forces to the optimizer as the analytic gradient
        (enables the portfolio's gradient arms and final polish).
    """

    def __init__(self, atoms: Atoms, *, budget: int = 20000, seed: int = 0,
                 pad: float = 3.0, use_forces: bool = True):
        if atoms.calc is None:
            raise ValueError("AnnealOptimizer needs atoms with a calculator attached")
        self.atoms = atoms
        self.budget = int(budget)
        self.seed = int(seed)
        self.pad = float(pad)
        self.use_forces = bool(use_forces)
        self.result: dict | None = None

    def _energy(self, x: np.ndarray) -> float:
        self.atoms.set_positions(np.asarray(x).reshape(-1, 3))
        return float(self.atoms.get_potential_energy())

    def _gradient(self, x: np.ndarray) -> np.ndarray:
        self.atoms.set_positions(np.asarray(x).reshape(-1, 3))
        return (-self.atoms.get_forces()).ravel()  # grad E = -F

    def run(self) -> Atoms:
        """Optimize and write the best geometry back into ``atoms``.

        Returns the same :class:`ase.Atoms` (now at the optimized geometry).
        """
        import anneal  # lazy: only needed when actually optimizing

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
        """Best energy found (after :meth:`run`)."""
        if self.result is None:
            raise RuntimeError("call run() first")
        return float(self.result["best_val"])

    @property
    def nevals(self) -> int:
        if self.result is None:
            return 0
        return int(self.result["n_evals"]) + int(self.result["n_grads"])


def atoms_from_con(path, index: int = -1, *, calc=None, cell=None, pbc=None) -> Atoms:
    """Read an EON ``.con`` file into an :class:`ase.Atoms` via readcon-core.

    Uses readcon's native ASE export (``read_con_as_ase``), so by default the
    cell is taken from the ``.con`` cellpar and element symbols are resolved for
    you. ``cell`` / ``pbc`` override those defaults when given -- handy when the
    ``.con`` cell is absent, or when a cluster needs a large non-periodic box.

    A frame that stored forces comes back with the ``SinglePointCalculator``
    ``to_ase`` attached, including the frame energy when that calculator
    carries one. That calculator stays unless ``calc`` is passed.

    Parameters
    ----------
    path:
        Path to the ``.con`` file.
    index:
        Frame index to return (default last frame).
    calc:
        Optional calculator to attach to the returned atoms.
    cell:
        Optional ASE cell override (anything ``Atoms.set_cell`` accepts).
    pbc:
        Optional periodic-boundary override (bool or 3-tuple).
    """
    import readcon  # lazy: readcon-core's Python module

    atoms = readcon.read_con_as_ase(str(path))[index]
    if cell is not None:
        atoms.set_cell(cell)
    if pbc is not None:
        atoms.set_pbc(pbc)
    if calc is not None:
        atoms.calc = calc
    return atoms


def atoms_to_con(path, atoms: Atoms) -> None:
    """Write ``atoms`` as one EON ``.con`` frame.

    ``ConFrame.from_ase`` then ``write_con`` with precision omitted and
    canonical left off. No corpus.
    """
    import readcon  # lazy: readcon-core's Python module

    frame = readcon.ConFrame.from_ase(atoms)
    readcon.write_con(str(path), [frame])


def atoms_to_calc(atoms: Atoms, calc) -> Atoms:
    """Attach ``calc`` to ``atoms`` and return it (fluent helper)."""
    atoms.calc = calc
    return atoms
