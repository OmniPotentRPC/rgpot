# MIT License
# Copyright 2023--present rgpot developers
"""F1 import check for a free-threaded wheel.

On a free-threaded interpreter, import every name in RGBIND_IMPORTS
(default: _rgbind_test) with RuntimeWarning as an error and require
sys._is_gil_enabled() is False. GIL builds exit 0 without asserting.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import warnings


def free_threaded() -> bool:
    gil = sysconfig.get_config_var("Py_GIL_DISABLED")
    if str(gil) in {"1", "True"}:
        return True
    flags = str(sysconfig.get_config_var("ABIFLAGS") or "")
    return "t" in flags


def main() -> int:
    raw = os.environ.get("RGBIND_IMPORTS", "_rgbind_test")
    names = [part for part in raw.replace(",", " ").split() if part]
    if not free_threaded():
        print("F1: GIL build, skip")
        return 0
    warnings.simplefilter("error", RuntimeWarning)
    for name in names:
        __import__(name)
    enabled = sys._is_gil_enabled()
    print(f"F1: sys._is_gil_enabled()={enabled} modules={names}")
    if enabled:
        print("F1 failed: GIL is enabled", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
