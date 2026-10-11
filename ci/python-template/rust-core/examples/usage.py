# MIT License
# Copyright 2023--present rgpot developers
"""Doc example: the rust-core demo dot product."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import rust_core_demo


def dot(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    """Return the fixed-order dot product."""
    return float(rust_core_demo.dot(left, right))
