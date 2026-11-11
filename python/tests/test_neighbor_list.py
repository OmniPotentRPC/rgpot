# MIT License
# Copyright 2023--present rgpot developers

"""Three-line neighbour list, the same shape as ASE's neighbor_list."""

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def test_neighbor_list_is_three_lines():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    i, j, d = rgpot.neighbor_list(positions, cell, 2.0)
    assert list(i) == [0]
    assert list(j) == [1]
    assert d[0] == pytest.approx(1.2)
