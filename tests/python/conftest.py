"""Pytest hooks for the free-threaded ASE suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Source tree: python/rgpot. A wheel install already has rgpot on the path.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_PYTHON = _ROOT / "python"
sys.path.insert(0, str(_HERE))
if (_PYTHON / "rgpot" / "__init__.py").is_file():
    sys.path.insert(0, str(_PYTHON))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "thread_unsafe: not safe under pytest-run-parallel",
    )
