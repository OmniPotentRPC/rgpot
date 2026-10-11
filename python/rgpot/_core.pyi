"""
rgpot Python bindings (nanobind): LJ, Fortran kernels, Metatomic dlopen, ExprPot, D3Pot, D4Pot, and XcKernel when the build flags are on. Evaluations release the thread (F4).
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

has_metatomic_dlopen: bool = False

has_cache: bool = False

has_expr: bool = False

has_dftd3: bool = False

has_dftd4: bool = False

has_xckernel: bool = False

has_fortran_pots: bool = True

def echo_f64(values: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]: ...

def evaluate_lj(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

class LJPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None: ...

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class SWPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class EDIPPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class LenoskyPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class TersoffPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class EAMAlPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class FeHePot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class CuH2Pot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class WaterHPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: Annotated[NDArray[numpy.int32], dict(order='C', device='cpu', writable=False)], box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """
