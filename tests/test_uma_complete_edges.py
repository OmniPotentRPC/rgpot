# Copyright (c) rgpot developers
"""The molecular edge list keeps a pair the 10 angstrom cutoff would drop."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from uma_edges import complete_ordered_edges  # noqa: E402

# Frame 0 of the attached Claisen endpoints. Raw numbers are angstrom:
# the nearest contact is 0.5576, and treating the file as bohr
# (factor 0.529177210903) makes that contact 0.295 angstrom. Neither is
# a bond, and the frame's longest pair is 3.30, so the failing dimer
# frame is not a uniform rescaling. Atoms 5 and 11 are moved equally
# along their joining line to the logged distance.
_FRAME0 = [
    (0.96601080, -0.33392482, 0.69680887),
    (0.27933941, -0.39005711, 0.64163902),
    (-0.08738951, -0.63637662, -0.00872396),
    (-0.37571458, 0.43525148, -0.51394545),
    (-0.80817203, 0.99043931, -0.67109528),
    (-0.63111841, -0.17400821, -0.24752542),
    (1.32236146, -0.48145900, 0.27432843),
    (1.21485645, -0.15499709, 1.17824275),
    (-0.06006874, -0.24234469, 1.07435274),
    (-0.36090676, -1.14831179, 0.10468814),
    (0.29610618, -0.73196832, -0.42999683),
    (-0.58501592, 1.45189783, -0.90637601),
    (-1.35391476, 0.96676073, -0.55911516),
    (0.18362638, 0.44909833, -0.63328183),
]
_TARGET = 10.197616276524268
_BOHR_TO_ANGSTROM = 0.529177210903


def _dist(a, b):
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def claisen_geometry():
    pos = [list(row) for row in _FRAME0]
    mid = [(pos[5][k] + pos[11][k]) / 2.0 for k in range(3)]
    step = [pos[11][k] - pos[5][k] for k in range(3)]
    length = math.sqrt(sum(v * v for v in step))
    direction = [v / length for v in step]
    half = _TARGET / 2.0
    pos[5] = [mid[k] - direction[k] * half for k in range(3)]
    pos[11] = [mid[k] + direction[k] * half for k in range(3)]
    return pos


def test_endpoint_units_are_not_a_hidden_bohr_scale():
    nearest = min(
        _dist(_FRAME0[i], _FRAME0[j])
        for i in range(14)
        for j in range(i + 1, 14)
    )
    assert abs(nearest - 0.557615) < 1e-3
    assert nearest * _BOHR_TO_ANGSTROM < 0.4
    longest = max(
        _dist(_FRAME0[i], _FRAME0[j])
        for i in range(14)
        for j in range(i + 1, 14)
    )
    assert longest < 4.0


def test_complete_list_keeps_the_pair_a_10_angstrom_cutoff_drops():
    pos = claisen_geometry()
    assert abs(_dist(pos[5], pos[11]) - _TARGET) < 1e-12
    neighbor, center, shifts = complete_ordered_edges(pos)
    assert len(neighbor) == 14 * 13
    assert len(center) == 14 * 13
    assert shifts[0] == (0.0, 0.0, 0.0)
    directed_below_10 = 0
    saw = False
    for c, n in zip(center, neighbor):
        d = _dist(pos[c], pos[n])
        if d < 10.0:
            directed_below_10 += 1
        if (c, n) == (5, 11) or (c, n) == (11, 5):
            assert d == _TARGET or abs(d - _TARGET) < 1e-12
            saw = True
    assert saw
    assert directed_below_10 == 180


if __name__ == "__main__":
    test_endpoint_units_are_not_a_hidden_bohr_scale()
    test_complete_list_keeps_the_pair_a_10_angstrom_cutoff_drops()
    print("ok")
