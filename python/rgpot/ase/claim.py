"""Compare-and-swap claim for one mutable session (F3).

The flag is set for the whole compute region. The mutex covers only the
flag update, so a second caller raises instead of waiting.
"""

from __future__ import annotations

import threading
from typing import Literal


class ConnectionClaim:
    """F3 claim. A second concurrent enter raises RuntimeError."""

    def __init__(self, type_name: str) -> None:
        self._type_name = type_name
        self._busy = False
        self._mu = threading.Lock()

    def __enter__(self) -> ConnectionClaim:
        with self._mu:
            if self._busy:
                raise RuntimeError(
                    f"{self._type_name} is in use by another thread"
                )
            self._busy = True
        return self

    def __exit__(self, *exc: object) -> Literal[False]:
        with self._mu:
            self._busy = False
        return False
