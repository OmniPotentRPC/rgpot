"""Threaded EAM-Al, Stillinger-Weber, and Tersoff rows.

asv times ``time_*`` only. ``setup`` builds the geometry and, for the
in-skin and warm rows, the first neighbour list. Each row is skipped
when ``rgpot.has_eam_al_openmp`` is false.

The thread count is the asv parameter ``threads``. ``RGPOT_OMP_THREADS``
overrides it: a comma-separated list, default ``8``.
"""

from __future__ import annotations

import ctypes
import math
import os

_HALF_SKIN = 0.355


def _thread_values() -> list[int]:
    raw = os.environ.get("RGPOT_OMP_THREADS", "8")
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise RuntimeError("RGPOT_OMP_THREADS is empty")
    return values


def _set_threads(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_DYNAMIC"] = "false"
    try:
        lib = ctypes.CDLL("libgomp.so.1")
    except OSError:
        return
    lib.omp_set_num_threads(threads)


def _require_openmp() -> None:
    import rgpot

    if not rgpot.has_eam_al_openmp:
        raise NotImplementedError("with_eam_al_openmp is off")


def _fcc(n: int, lattice: float = 4.05):
    import numpy as np

    basis = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    ncell = int(math.ceil((n / 4.0) ** (1.0 / 3.0)))
    coords = []
    for ix in range(ncell):
        for iy in range(ncell):
            for iz in range(ncell):
                for site in basis:
                    if len(coords) >= n:
                        break
                    coords.append((ix + site) * lattice)
                if len(coords) >= n:
                    break
            if len(coords) >= n:
                break
        if len(coords) >= n:
            break
    pos = np.ascontiguousarray(coords, dtype=np.float64)
    types = np.full(n, 13, dtype=np.int32)
    edge = ncell * lattice
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = box[1, 1] = box[2, 2] = edge
    return pos, types, box


def _diamond(n: int, lattice: float = 5.431):
    import numpy as np

    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ],
        dtype=np.float64,
    )
    side = int(round((n / 8.0) ** (1.0 / 3.0)))
    coords = []
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for site in basis:
                    atom = len(coords) + 1
                    xyz = (ix + site) * lattice
                    xyz[0] += 0.02 * math.sin(atom)
                    coords.append(xyz)
    pos = np.ascontiguousarray(coords, dtype=np.float64)
    types = np.full(pos.shape[0], 14, dtype=np.int32)
    edge = side * lattice
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = box[1, 1] = box[2, 2] = edge
    return pos, types, box


class _Row:
    params = ([4096, 32768], _thread_values())
    param_names = ["atoms", "threads"]

    def _geometry(self, atoms: int):
        raise NotImplementedError

    def _pot(self):
        raise NotImplementedError

    def setup(self, atoms: int, threads: int) -> None:
        _require_openmp()
        _set_threads(threads)
        base, types, box = self._geometry(atoms)
        self.base = base
        self.types = types
        self.box = box
        self.pos = base.copy()
        self.step = 0
        self.pot = self._pot()
        self.pot(self.base, self.types, self.box)


class _Eam(_Row):
    def _geometry(self, atoms: int):
        return _fcc(atoms)

    def _pot(self):
        import rgpot

        return rgpot.EAMAlPot()


class EamAlCold(_Eam):
    """One rebuilt list. The shift is larger than the skin."""

    def time_cold(self, atoms: int, threads: int) -> None:
        del atoms, threads
        self.step += 1
        self.pos[:, :] = self.base
        self.pos[:, 0] += 1.0 + 0.01 * (self.step % 20)
        self.pot(self.pos, self.types, self.box)


class EamAlSkin(_Eam):
    """One in-skin step. The displacement stays under half the skin."""

    def time_skin(self, atoms: int, threads: int) -> None:
        del atoms, threads
        self.step += 1
        self.pos[:, :] = self.base
        self.pos[:, 0] += 0.02 * ((self.step % 7) + 1)
        if float(self.pos[0, 0] - self.base[0, 0]) >= _HALF_SKIN:
            raise RuntimeError("in-skin row left the skin")
        self.pot(self.pos, self.types, self.box)


class EamAlMdMix(_Eam):
    """One rebuild and six in-skin steps, the mix of a 300 K run."""

    def time_md_mix(self, atoms: int, threads: int) -> None:
        del atoms, threads
        self.pos[:, :] = self.base
        self.pos[:, 0] += 1.0
        self.pot(self.pos, self.types, self.box)
        for _ in range(6):
            self.pos[:, 0] += 0.04
            self.pot(self.pos, self.types, self.box)


class SwWarm(_Row):
    """Identical-position call. The neighbour list is already built."""

    def _geometry(self, atoms: int):
        return _diamond(atoms)

    def _pot(self):
        import rgpot

        return rgpot.SWPot()

    def setup(self, atoms: int, threads: int) -> None:
        super().setup(atoms, threads)
        self.pot(self.base, self.types, self.box)

    def time_warm(self, atoms: int, threads: int) -> None:
        del atoms, threads
        self.pot(self.base, self.types, self.box)


class TersoffWarm(SwWarm):
    def _pot(self):
        import rgpot

        return rgpot.TersoffPot()
