# MIT License
# Copyright 2023--present rgpot developers
"""Hypothesis checks for array validation and the F7 table (T4)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import _rgbind_test

pytest.importorskip("hypothesis")

floats = st.floats(width=64, allow_nan=True, allow_infinity=True, allow_subnormal=True)


@given(st.lists(floats, min_size=0, max_size=8))
@settings(max_examples=40, deadline=None)
def test_echo_round_trip(values: list[float]) -> None:
    arr = np.array(values, dtype=np.float64)
    out = _rgbind_test.echo_f64(arr)
    assert out.tobytes() == arr.tobytes()


@given(
    st.integers(min_value=0, max_value=6),
    st.integers(min_value=0, max_value=6),
    st.integers(min_value=0, max_value=6),
    st.integers(min_value=0, max_value=6),
)
@settings(max_examples=30, deadline=None)
def test_shape_either_accepts_or_names_both(rows: int, cols: int, erows: int, ecols: int) -> None:
    arr = np.zeros((rows, cols), dtype=np.float64)
    if (rows, cols) == (erows, ecols):
        _rgbind_test.check_f64_2d(arr, erows, ecols, "positions")
        return
    with pytest.raises(ValueError) as exc:
        _rgbind_test.check_f64_2d(arr, erows, ecols, "positions")
    text = str(exc.value)
    assert "positions" in text
    assert f"({erows}, {ecols})" in text
    assert f"({rows}, {cols})" in text


@given(st.integers(min_value=0, max_value=11), st.text(max_size=20))
@settings(max_examples=25, deadline=None)
def test_status_kind_stable(code: int, last: str) -> None:
    kind, message = _rgbind_test.map_status(code, last, "Calc")
    again, message2 = _rgbind_test.map_status(code, last, "Calc")
    assert kind == again
    assert message == message2
    assert kind in {
        "ok",
        "value_error",
        "type_error",
        "not_converged",
        "callback",
        "numerical",
        "runtime_error",
        "not_implemented",
    }
