# MIT License
# Copyright 2023--present rgpot developers

"""Neighbour lists in the shape of ASE's neighbor_list."""

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def test_three_line_half_list():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    i, j, d = rgpot.neighbor_list(positions, cell, 2.0)
    assert list(i) == [0]
    assert list(j) == [1]
    assert d[0] == pytest.approx(1.2)
    assert i.base is not None
    assert j.base is not None
    assert d.base is not None


def test_quantity_selector_returns_one_array():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    d = rgpot.neighbor_list(positions, cell, 2.0, quantities="d")
    assert d.shape == (1,)
    assert d[0] == pytest.approx(1.2)


def test_displacement_and_shift():
    positions = np.array([[0.0, 0.0, 0.0], [7.5, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    i, j, D, S = rgpot.neighbor_list(positions, cell, 2.0, quantities="ijDS")
    assert list(i) == [0]
    assert list(j) == [1]
    assert D.shape == (1, 3)
    assert D[0, 0] == pytest.approx(-0.5)
    assert S.shape == (1, 3)
    assert list(S[0]) == [-1, 0, 0]


def test_full_list_offsets_are_sorted_csr():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.2, 1.1, 0.0]], dtype=np.float64)
    cell = np.diag([10.0, 10.0, 10.0])
    i, j, offsets = rgpot.neighbor_list(positions, cell, 2.0, quantities="ij", full=True, offsets=True)
    assert offsets.shape == (4,)
    assert int(offsets[0]) == 0
    assert int(offsets[-1]) == len(i)
    for atom in range(3):
        block = slice(int(offsets[atom]), int(offsets[atom + 1]))
        assert list(i[block]) == [atom] * (int(offsets[atom + 1]) - int(offsets[atom]))
        assert list(j[block]) == sorted(j[block])
    assert len(i) == 6


def test_unknown_quantity_is_rejected():
    positions = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    cell = np.diag([8.0, 8.0, 8.0])
    with pytest.raises(ValueError):
        rgpot.neighbor_list(positions, cell, 2.0, quantities="ijx")
