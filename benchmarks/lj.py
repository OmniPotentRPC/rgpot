"""Lennard-Jones force timings and memray peaks for the Python binding."""

from __future__ import annotations

import numpy as np

import rgpot


def _geometry(atoms: int):
    rng = np.random.default_rng(1)
    positions = np.ascontiguousarray(rng.random((atoms, 3)) * 10.0, dtype=np.float64)
    atom_types = np.zeros(atoms, dtype=np.int32)
    box = np.ascontiguousarray(np.eye(3) * 40.0, dtype=np.float64)
    return positions, atom_types, box


class LennardJones:
    """Wall time of one force evaluation.

    ``warmup_time`` discards calls before the first recorded sample.
    ``evaluate_lj`` builds a new calculator on every call, and the
    coordinate arrays are not written, so a warm call cannot change the
    geometry or the pair list of the timed call. ``LJPot`` is one
    object: the setup call fills its pair list, and every timed call
    repeats that same geometry. The positions, types, and cell are the
    arrays created in setup; the warm call does not replace them.
    """

    params = [128, 512]
    param_names = ["atoms"]
    rounds = 10
    repeat = 1
    sample_time = 0.05
    warmup_time = 0.3
    min_run_count = 1
    timeout = 180

    def setup(self, atoms: int) -> None:
        self.positions, self.atom_types, self.box = _geometry(atoms)
        self.pot = rgpot.LJPot()
        rgpot.evaluate_lj(self.positions, self.atom_types, self.box)
        self.pot(self.positions, self.atom_types, self.box)

    def time_evaluate_lj(self, atoms: int) -> None:
        rgpot.evaluate_lj(self.positions, self.atom_types, self.box)

    def time_ljpot_call(self, atoms: int) -> None:
        self.pot(self.positions, self.atom_types, self.box)


class LennardJonesMemory:
    """Peak bytes from asv_bench_memray (methods named ray_*)."""

    params = [128, 512]
    param_names = ["atoms"]
    rounds = 2
    number = 1
    repeat = 1
    timeout = 180

    def ray_evaluate_lj(self, atoms: int) -> None:
        positions, atom_types, box = _geometry(atoms)
        rgpot.evaluate_lj(positions, atom_types, box)

    def ray_ljpot_call(self, atoms: int) -> None:
        positions, atom_types, box = _geometry(atoms)
        pot = rgpot.LJPot()
        pot(positions, atom_types, box)
