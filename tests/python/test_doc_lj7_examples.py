# Copyright 2023--present rgpot developers

"""The LJ7 tutorial states the numbers a current ASE run prints.

``docs/orgmode/tutorials/molecular-global-optimization.org`` shows an
RgpotRpcCalculator on a 7-atom Lennard-Jones cluster. That calculator
passes the rank-3 cell through even when ``pbc`` is left false, which is
the periodic box ``RgpotCalculator("lj")`` uses when ``pbc`` is true.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from rgpot.ase import RgpotCalculator
from scipy.optimize import dual_annealing

_TUTORIAL = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "orgmode"
    / "tutorials"
    / "molecular-global-optimization.org"
)
_N = 7
_BOX = [40.0, 40.0, 40.0]


def _start() -> np.ndarray:
    return np.random.RandomState(1).uniform(-2.3, 2.3, size=(_N, 3))


def _atoms(positions: np.ndarray) -> Atoms:
    atoms = Atoms(numbers=[0] * _N, positions=positions, cell=_BOX, pbc=True)
    atoms.calc = RgpotCalculator("lj")
    return atoms


def _tutorial() -> str:
    return _TUTORIAL.read_text(encoding="utf-8")


def test_unlimited_lbfgs_matches_the_tutorial() -> None:
    text = _tutorial()
    match = re.search(
        r"converges in (\d+) steps to a minimum at (-?\d+\.\d+)",
        text,
    )
    assert match is not None
    atoms = _atoms(_start())
    opt = LBFGS(atoms, logfile=None)
    assert opt.run(fmax=1e-3)
    assert opt.get_number_of_steps() == int(match.group(1))
    assert f"{atoms.get_potential_energy():.4f}" == match.group(2)


def test_capped_lbfgs_matches_the_tutorial_table() -> None:
    text = _tutorial()
    match = re.search(
        r"ASE L-BFGS \(local\), 500 steps\s+\|\s+(-?\d+\.\d+)",
        text,
    )
    assert match is not None
    atoms = _atoms(_start())
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=1e-3, steps=500)
    assert opt.get_number_of_steps() == 500
    assert f"{atoms.get_potential_energy():.6f}" == match.group(1)


def test_dual_annealing_matches_the_tutorial_table() -> None:
    text = _tutorial()
    match = re.search(
        r"scipy ~dual_annealing~ \(global\)\s+\|\s+(-?\d+\.\d+)\s+\|\s+(\d+)",
        text,
    )
    assert match is not None
    start = _start()
    scratch = _atoms(start)
    dim = _N * 3

    def energy(x: np.ndarray) -> float:
        scratch.set_positions(np.asarray(x).reshape(_N, 3))
        return float(scratch.get_potential_energy())

    def jac(x: np.ndarray) -> np.ndarray:
        scratch.set_positions(np.asarray(x).reshape(_N, 3))
        return (-scratch.get_forces()).ravel()

    result = dual_annealing(
        energy,
        list(zip([-3.0] * dim, [3.0] * dim)),
        x0=np.asarray(start).ravel(),
        maxiter=400,
        seed=11,
        minimizer_kwargs={"method": "L-BFGS-B", "jac": jac},
    )
    assert f"{float(result.fun):.6f}" == match.group(1)
    assert int(result.nfev) == int(match.group(2))
