"""Wall time of one force call on the orthorhombic fast paths.

``setup`` loads the geometry and warms the pair cache. asv does not time
``setup``. Each ``time_force_call`` is one ``forceImpl``.

The shared library is the meson target ``asv_fastpath``, built with
``-Dwith_tests=true``. Set ``RGPOT_ASV_LIB`` to ``libasv_fastpath.so``
when the build directory is not ``build/``.

LJ38, LJ75 and LJ150 sit inside the cutoff, so the warm walk is every
pair. The periodic lattice is wide enough that linkcell returns a short
list. Pt warm folds the cached list with minimage. Pt cold moves the
island by 1 Angstrom on each call so the list is built again. Water plus
H wraps every leg with minimage. Optbench's own asv times the search, not
these lists.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_LIB = None

LJ38 = 0
LJ75 = 1
LJ150 = 2
PT_WARM = 3
PT_COLD = 4
WATER = 5
LJ_LINKCELL = 6


def _library() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    env = os.environ.get("RGPOT_ASV_LIB")
    root = Path(__file__).resolve().parents[1]
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            root / "build" / "libasv_fastpath.so",
            root / "build" / "CppCore" / "libasv_fastpath.so",
        ]
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError(
            "libasv_fastpath.so was not found. Build with -Dwith_tests=true "
            "or set RGPOT_ASV_LIB."
        )
    lib = ctypes.CDLL(str(path))
    lib.rgpot_asv_prepare.argtypes = [ctypes.c_int]
    lib.rgpot_asv_prepare.restype = ctypes.c_int
    lib.rgpot_asv_force.argtypes = [ctypes.c_int]
    lib.rgpot_asv_force.restype = ctypes.c_double
    lib.rgpot_asv_last_error.argtypes = []
    lib.rgpot_asv_last_error.restype = ctypes.c_char_p
    _LIB = lib
    return lib


def _prepare(kind: int) -> None:
    lib = _library()
    if lib.rgpot_asv_prepare(kind) != 0:
        err = lib.rgpot_asv_last_error()
        text = err.decode() if err else "rgpot_asv_prepare failed"
        raise RuntimeError(text)


def _force(kind: int) -> None:
    _library().rgpot_asv_force(kind)


class Lj38:
    def setup(self):
        _prepare(LJ38)

    def time_force_call(self):
        _force(LJ38)


class Lj75:
    def setup(self):
        _prepare(LJ75)

    def time_force_call(self):
        _force(LJ75)


class Lj150:
    def setup(self):
        _prepare(LJ150)

    def time_force_call(self):
        _force(LJ150)


class PtIslandWarm:
    def setup(self):
        _prepare(PT_WARM)

    def time_force_call(self):
        _force(PT_WARM)


class PtIslandCold:
    def setup(self):
        _prepare(PT_COLD)

    def time_force_call(self):
        _force(PT_COLD)


class WaterH:
    def setup(self):
        _prepare(WATER)

    def time_force_call(self):
        _force(WATER)


class LjPeriodicLinkcell:
    def setup(self):
        _prepare(LJ_LINKCELL)

    def time_force_call(self):
        _force(LJ_LINKCELL)
