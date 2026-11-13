"""Wall time of one force call on the orthorhombic fast paths.

``setup`` loads the geometry and warms the pair cache. asv does not time
``setup``. Each ``time_force_call`` is one call on the installed
extension: ``LJClusterPot``, ``MorsePot``, ``WaterHPot``, or ``LJPot``.
Those classes call the force implementations in the installed
``librgpot``. The test library is not loaded.

LJ38, LJ75 and LJ150 sit inside the cutoff, so the warm walk is every
pair. The periodic lattice is wide enough that linkcell returns a short
list. Pt warm folds the cached list with minimage. Pt cold writes a
one-angstrom shift. Setup clears the step before every repeat, so the
shift repeats and the pair cache hits. The list is not built again on
each call. Water plus H wraps every leg with minimage.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

_RECORDED = False

_DATA = Path(__file__).resolve().parent / "data"


def _record(path: Path) -> None:
    """Write the loaded library checksum once per process.

    asv calls setup for every repeat inside a sample. Hashing on each
    of those calls buries the round record. Each round is its own
    process, so one line remains for that round.
    """
    global _RECORDED
    if _RECORDED:
        return
    _RECORDED = True
    env = os.environ.get("ASV_ENV_DIR")
    if not env:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    commit = os.environ.get("ASV_COMMIT", "")
    line = f"{commit}\t{path}\t{digest}\n"
    with (Path(env) / "lib-load.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def _mapped_librgpot() -> Path:
    env = os.environ.get("ASV_ENV_DIR")
    if not env:
        raise RuntimeError("ASV_ENV_DIR is unset")
    root = Path(env).resolve()
    needle = "librgpot.so"
    for line in Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace").splitlines():
        if needle not in line:
            continue
        text = line.rsplit(" ", 1)[-1]
        if not text.startswith("/"):
            continue
        path = Path(text).resolve()
        if not path.is_file():
            continue
        if path != root and root not in path.parents:
            raise RuntimeError(f"mapped {path} is outside {root}")
        return path
    raise RuntimeError("librgpot.so is not mapped after importing rgpot")


def _load_con(path: Path, atomic_number: int):
    import numpy as np

    rest = path.read_text(encoding="utf-8").split("\n", 1)[1]
    head, _, tail = rest.partition("Coordinates of Component")
    nums = head.split()
    nat = int(float(nums[7]))
    coords = []
    for line in tail.splitlines()[1:]:
        bits = line.split()
        if len(bits) < 3:
            continue
        coords.append((float(bits[0]), float(bits[1]), float(bits[2])))
        if len(coords) == nat:
            break
    if len(coords) != nat:
        raise RuntimeError(f"{path.name} has {len(coords)} atoms, expected {nat}")
    pos = np.ascontiguousarray(coords, dtype=np.float64)
    types = np.full(nat, atomic_number, dtype=np.int32)
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = float(nums[0])
    box[1, 1] = float(nums[1])
    box[2, 2] = float(nums[2])
    return pos, types, box


def _load_xyz(path: Path):
    import numpy as np

    coords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        bits = line.split()
        if len(bits) < 3:
            continue
        coords.append((float(bits[0]), float(bits[1]), float(bits[2])))
    pos = np.ascontiguousarray(coords, dtype=np.float64)
    types = np.ones(pos.shape[0], dtype=np.int32)
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = box[1, 1] = box[2, 2] = 100.0
    return pos, types, box


def _lattice():
    import numpy as np

    coords = [
        ((i + 0.5) * 5.0, (j + 0.5) * 5.0, (k + 0.5) * 5.0)
        for i in range(5)
        for j in range(5)
        for k in range(5)
    ]
    pos = np.ascontiguousarray(coords, dtype=np.float64)
    types = np.ones(pos.shape[0], dtype=np.int32)
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = box[1, 1] = box[2, 2] = 25.0
    return pos, types, box


def _water():
    import numpy as np

    nmol = 10
    nat = 3 * nmol + 1
    pos = np.zeros((nat, 3), dtype=np.float64)
    types = np.ones(nat, dtype=np.int32)
    roh = 0.9572
    half = 0.5 * 104.52 * math.acos(-1.0) / 180.0
    armx = roh * math.sin(half)
    army = roh * math.cos(half)
    for mol in range(nmol):
        ox = 3.0 + (mol % 5) * 3.2
        oy = 3.0 + (mol // 5) * 4.0
        h1 = 2 * mol
        h2 = 2 * mol + 1
        oxygen = 2 * nmol + mol
        pos[h1] = (ox + armx, oy + army, 3.0)
        pos[h2] = (ox - armx, oy + army, 3.0)
        pos[oxygen] = (ox, oy, 3.0)
        types[oxygen] = 8
    pos[nat - 1] = (8.0, 8.0, 4.2)
    box = np.zeros((3, 3), dtype=np.float64)
    box[0, 0] = box[1, 1] = box[2, 2] = 24.0
    return np.ascontiguousarray(pos, dtype=np.float64), types, box


def _wobble(base, phase: float):
    import numpy as np

    out = np.array(base, dtype=np.float64, copy=True, order="C")
    amp = 0.02
    idx = np.arange(out.shape[0], dtype=np.float64)
    out[:, 0] += amp * np.sin(0.37 * phase + idx)
    out[:, 1] += amp * np.sin(0.41 * phase + 1.3 * idx)
    out[:, 2] += amp * np.sin(0.53 * phase + 2.1 * idx)
    return np.ascontiguousarray(out, dtype=np.float64)


class _ForceCall:
    def _prepare(self, pot, base, types, box, cold: bool) -> None:
        import rgpot

        _record(_mapped_librgpot())
        self.pot = pot
        self.base = base
        self.types = types
        self.box = box
        self.cold = cold
        self.step = 0
        # One buffer for every sample. The warm wobble is the same
        # coordinates each call, so it is filled here. The cold row
        # overwrites that buffer in place.
        self.pos = base.copy()
        self.pot(self.pos, self.types, self.box)
        if not cold:
            self.pos = _wobble(self.base, 1.0)
            self.pot(self.pos, self.types, self.box)
        del rgpot

    def time_force_call(self) -> None:
        if self.cold:
            self.step += 1
            self.pos[:, :] = self.base
            self.pos[:, 0] += float(self.step)
        self.pot(self.pos, self.types, self.box)


class Lj38(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _load_con(_DATA / "lj38.con", 1)
        self._prepare(rgpot.LJClusterPot(cutoff=15.0), base, types, box, False)


class Lj75(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _load_xyz(_DATA / "lj75.xyz")
        self._prepare(rgpot.LJClusterPot(cutoff=15.0), base, types, box, False)


class Lj150(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _load_xyz(_DATA / "lj150.xyz")
        self._prepare(rgpot.LJClusterPot(cutoff=15.0), base, types, box, False)


class PtIslandWarm(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _load_con(_DATA / "pt-island.con", 78)
        self._prepare(rgpot.MorsePot(), base, types, box, False)


class PtIslandCold(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _load_con(_DATA / "pt-island.con", 78)
        self._prepare(rgpot.MorsePot(), base, types, box, True)


class WaterH(_ForceCall):
    def setup(self) -> None:
        import rgpot

        if rgpot.WaterHPot is None:
            raise RuntimeError("this build has no WaterHPot")
        base, types, box = _water()
        self._prepare(rgpot.WaterHPot(), base, types, box, False)


class LjPeriodicLinkcell(_ForceCall):
    def setup(self) -> None:
        import rgpot

        base, types, box = _lattice()
        self._prepare(rgpot.LJPot(cutoff=6.0), base, types, box, False)
