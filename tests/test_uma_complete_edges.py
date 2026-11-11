# Copyright 2023--present rgpot developers
"""The molecular edge list keeps a pair the 10 angstrom cutoff would drop."""

from __future__ import annotations

import math
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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


def _envelope_energy(pos, center, neighbor, cutoff):
    c2 = cutoff * cutoff
    energy = 0.0
    for c, n in zip(center, neighbor):
        d = _dist(pos[c], pos[n])
        r2 = d * d
        if r2 < c2:
            x = r2 / c2
            energy += (1.0 - x) * (1.0 - x)
    return energy


def _traced_molecular_edges(pos, molecular_box, backbone_cutoff):
    """Edges a molecular export traces.

    The current exporter lists every ordered pair. Commit 96e91006 set
    the trace cutoff to 0.4 times the box and kept pairs inside it.
    """
    try:
        from uma_edges import molecular_edges
    except ImportError:
        molecular_edges = None
    if not (float(backbone_cutoff) > 0.0):
        raise ValueError("backbone cutoff must be positive")
    if molecular_edges is not None:
        neighbor, center, _shifts = molecular_edges(pos)
        return neighbor, center
    # Commit 96e91006 traced a molecular box at 0.4 times the box side.
    cutoff = 0.4 * float(molecular_box)
    neighbor = []
    center = []
    for c in range(len(pos)):
        for n in range(len(pos)):
            if c == n:
                continue
            if _dist(pos[c], pos[n]) < cutoff:
                neighbor.append(n)
                center.append(c)
    return neighbor, center


def test_claisen_frame_has_182_edges_and_finite_energy():
    pos = claisen_geometry()
    assert abs(_dist(pos[5], pos[11]) - _TARGET) < 1e-12
    neighbor, center = _traced_molecular_edges(pos, 25.0, 6.0)
    assert len(neighbor) == 14 * 13, f"edge count {len(neighbor)}"
    assert len(center) == 14 * 13
    energy = _envelope_energy(pos, center, neighbor, 6.0)
    assert math.isfinite(energy)


def test_package_records_the_backbone_cutoff():
    from uma_edges import package_cutoffs

    backbone = 6.0
    box = 25.0
    cutoff, model_cutoff = package_cutoffs(backbone, box)
    assert cutoff == backbone
    assert model_cutoff == backbone
    assert cutoff != 0.4 * box
    assert model_cutoff != 0.4 * box


def test_complete_list_keeps_the_pair_a_10_angstrom_cutoff_drops():
    from uma_edges import complete_ordered_edges

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
    try:
        test_claisen_frame_has_182_edges_and_finite_energy()
    except AssertionError as exc:
        print(f"FAIL test_claisen_frame_has_182_edges_and_finite_energy: {exc}")
        raise SystemExit(1)
    print("PASS test_claisen_frame_has_182_edges_and_finite_energy edges=182 energy=finite")
    try:
        test_package_records_the_backbone_cutoff()
    except Exception as exc:
        print(f"FAIL test_package_records_the_backbone_cutoff: {type(exc).__name__}: {exc}")
        raise SystemExit(1)
    print("PASS test_package_records_the_backbone_cutoff cutoff=6 model_cutoff=6")
