"""Deprecated import path for the Cap'n Proto ASE calculator.

Use :class:`rgpot.ase.RgpotRpcCalculator`.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_python = Path(__file__).resolve().parents[3] / "python"
if _python.is_dir() and str(_python) not in sys.path:
    sys.path.insert(0, str(_python))

warnings.warn(
    "ase_calculator.RgpotCalculator is deprecated; "
    "use rgpot.ase.RgpotRpcCalculator",
    DeprecationWarning,
    stacklevel=2,
)

from rgpot.ase.rpc import RgpotRpcCalculator as RgpotCalculator  # noqa: E402

__all__ = ["RgpotCalculator"]
