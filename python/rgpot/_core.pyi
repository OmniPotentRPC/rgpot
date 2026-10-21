"""
rgpot Python bindings (nanobind): LJ, Fortran kernels, Metatomic dlopen, ExprPot, D3Pot, D4Pot, and XcKernel when the build flags are on. Evaluations release the thread (F4).
"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


class NumericalError(ArithmeticError):
    pass

has_metatomic_dlopen: bool = True

has_cache: bool = False

has_expr: bool = True

has_dftd3: bool = True

has_dftd4: bool = True

has_xckernel: bool = True

has_fortran_pots: bool = True

has_ipi: bool = True

def echo_f64(values: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]: ...

def evaluate_lj(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

def evaluate_metatomic_dlopen(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], model_path: str, engine_path: str = '', device: str = 'cpu') -> tuple:
    """
    Evaluate forces via MetatomicDlopen (dlopen libmetatomic_engine.so). engine_path may be empty to use RGPOT_METATOMIC_ENGINE / package-bundled multi-ABI engine under rgpot/lib/torch-X.Y/. Each call builds its own frontend (F4).
    """

class LJPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None: ...

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
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

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class D3Pot:
    def __init__(self, damping: str = 'bj', functional: str = 'pbe', atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class D4Pot:
    def __init__(self, functional: str = 'pbe', charge: float = 0.0, atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class ExprPot:
    def __init__(self, expression: str, terms: dict) -> None:
        """
        expression is a Lepton string; terms maps names to 'lj'/'morse'/'d3'/'d4' or a dict with kind plus D3/D4 fields.
        """

    @property
    def expression(self) -> str: ...

    def d_energy_d_term(self, name: str) -> float: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class IPIPot:
    def __init__(self, address: str, timeout_s: float = 30.0) -> None:
        """
        i-PI socket server. address is unix:/absolute/path, unix:<name> (binds /tmp/ipi_<name>), tcp:<host>:<port>, or tcp:<port>. The driver connects and returns forces. Wire units are bohr and hartree. timeout_s covers connecting and one force exchange.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def stress(self) -> object:
        """
        Cauchy stress from the last call, shape (3, 3), in eV/Angstrom^3. None before a successful call.
        """

    @property
    def endpoint(self) -> str: ...

class XcKernel:
    def __init__(self, name: str) -> None: ...

    @staticmethod
    def catalog() -> list[str]: ...

    @property
    def name(self) -> str: ...

    def scal_names(self) -> list[str]: ...

    def contract(self, chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object | None = None, scal: dict, lapl_chi: object | None = None, hess_chi: object | None = None) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @staticmethod
    def fields_from_density(chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object, density: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], lapl_chi: object | None = None, hess_chi: object | None = None) -> dict: ...
