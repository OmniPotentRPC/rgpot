"""rgbind policy tests (F1-F9)."""

from collections.abc import Callable
from typing import Annotated

import numpy
from numpy.typing import NDArray


RGBIND_VERSION: int = 1

class NumericalError(ArithmeticError):
    pass

class Session:
    def __init__(self) -> None: ...

    def work(self, hold_ms: int = 1) -> int:
        """Claim the session, hold, and release (F3, F4)."""

    def work_throw(self) -> None:
        """Claim, throw, and release via RAII (F3)."""

def call_on_worker(fn: Callable) -> None:
    """Run fn on a foreign thread through the F5 trampoline."""

def call_on_main(fn: Callable, check_signals: bool = False) -> int:
    """Run fn on this thread through the F5 trampoline."""

def schedule_canonical(rows: list, n_workers: int) -> str:
    """Canonical F5/F8 schedule. Independent of n_workers."""

def check_f64_1d(arr: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], n: int, name: str) -> None: ...

def check_f64_2d(arr: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], rows: int, cols: int, name: str) -> None: ...

def check_f64_3d(arr: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], d0: int, d1: int, d2: int, name: str) -> None: ...

def echo_f64(arr: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]:
    """Byte-copy a float64 array into a capsule-owned numpy array (F6)."""

def map_status(code: int, last_error: str = '', type_name: str = 'Object') -> tuple[str, str]:
    """Return (kind, message) for an F7 status code."""

def raise_status(code: int, last_error: str = '', type_name: str = 'Object') -> bool:
    """Raise the F7 exception, or return False when not converged."""
