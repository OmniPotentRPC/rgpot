# MIT License
# Copyright 2023--present rgpot developers
"""Smoke test for the rust-core demo wheel."""

from __future__ import annotations

import sys
import sysconfig

import numpy as np
import pytest

import rust_core_demo


def test_dot() -> None:
    left = np.array([1.0, 2.0, 3.0])
    right = np.array([4.0, 5.0, 6.0])
    assert rust_core_demo.dot(left, right) == 32.0


def test_shape() -> None:
    with pytest.raises(ValueError, match=r"expected shape"):
        rust_core_demo.dot(np.array([1.0, 2.0]), np.array([1.0]))


def test_scalar_rank() -> None:
    vector = np.array([1.0])
    scalar = np.array(1.0)
    with pytest.raises(ValueError, match=r"^a:"):
        rust_core_demo.dot(scalar, vector)
    with pytest.raises(ValueError, match=r"^b:"):
        rust_core_demo.dot(vector, scalar)


def test_gil_disabled_on_free_threaded() -> None:
    gil = sysconfig.get_config_var("Py_GIL_DISABLED")
    flags = str(sysconfig.get_config_var("ABIFLAGS") or "")
    if str(gil) in {"1", "True"} or "t" in flags:
        assert sys._is_gil_enabled() is False
