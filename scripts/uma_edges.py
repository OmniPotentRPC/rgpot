# Copyright (c) rgpot developers
"""Ordered pair lists shared by the UMA exporter and its tests.

A molecular AOTI package is traced at n(n-1) directed edges. The list
is every ordered pair, sorted by (center, distance, neighbor), with a
zero cell shift. The sort matches UmaPot's complete_ordered_edges.
"""

from __future__ import annotations

import math


def complete_ordered_edges(pos):
    """Return (neighbor, center) index lists and zero shifts.

    ``pos`` is a sequence of length-3 coordinates. Self pairs are
    omitted. The order is (center, distance, neighbor).
    """
    n = len(pos)
    rows = []
    for center in range(n):
        cx, cy, cz = pos[center]
        for neighbor in range(n):
            if neighbor == center:
                continue
            nx, ny, nz = pos[neighbor]
            dx = nx - cx
            dy = ny - cy
            dz = nz - cz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            rows.append((center, dist, neighbor))
    rows.sort()
    neighbor_index = [row[2] for row in rows]
    center_index = [row[0] for row in rows]
    shifts = [(0.0, 0.0, 0.0) for _ in rows]
    return neighbor_index, center_index, shifts
