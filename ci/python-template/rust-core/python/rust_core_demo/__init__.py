# MIT License
# Copyright 2023--present rgpot developers
"""rust-core-demo: a fixed-order dot product behind a C ABI."""

from __future__ import annotations

from rust_core_demo._core import NumericalError, dot

__all__ = ["NumericalError", "dot"]
