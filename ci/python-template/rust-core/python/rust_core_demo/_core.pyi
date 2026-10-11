"""rust-core-demo: a fixed-order dot product behind a C ABI."""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

def dot(a: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], b: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> float:
    """Dot product of two float64 vectors. The sum is index order."""
