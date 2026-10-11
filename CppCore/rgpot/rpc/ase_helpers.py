#!/usr/bin/env python3
"""Ergonomic ASE helpers for the rgpot + anneal + readcon stack.

These wrap the lower-level pieces so a user works in plain ASE terms:

* :class:`AnnealRunner` -- a driver that is not an ``ase.optimize.Optimizer``.
  Construct it with an :class:`ase.Atoms` that already has a calculator, call
  :meth:`run`, and the atoms are relaxed in place by anneal's global portfolio
  optimizer using the calculator's energy and forces.
  :class:`AnnealOptimizer` is the deprecated name of the same driver.

* :func:`atoms_from_con` / :func:`atoms_to_calc` -- load atomic structures from
  EON ``.con`` files via ``readcon`` (readcon-core's Python module) straight into
  an :class:`ase.Atoms`, optionally attaching a calculator.

The heavy dependencies (``anneal``, ``readcon``) are imported lazily so importing
this module only needs ASE + numpy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ase import Atoms

_python = Path(__file__).resolve().parents[3] / "python"
if _python.is_dir() and str(_python) not in sys.path:
    sys.path.insert(0, str(_python))

from rgpot.ase.anneal import AnnealOptimizer, AnnealRunner  # noqa: E402


def atoms_from_con(path, index: int = -1, *, calc=None, cell=None, pbc=None) -> Atoms:
    """Read an EON ``.con`` file into an :class:`ase.Atoms` via readcon-core.

    Uses readcon's native ASE export (``read_con_as_ase``), so by default the
    cell is taken from the ``.con`` cellpar and element symbols are resolved for
    you. ``cell`` / ``pbc`` override those defaults when given -- handy when the
    ``.con`` cell is absent, or when a cluster needs a large non-periodic box.

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


def atoms_to_calc(atoms: Atoms, calc) -> Atoms:
    """Attach ``calc`` to ``atoms`` and return it (fluent helper)."""
    atoms.calc = calc
    return atoms
