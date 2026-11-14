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

has_eam_table: bool = False

def echo_f64(values: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> NDArray[numpy.float64]: ...

def evaluate_lj(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

def evaluate_metatomic_dlopen(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], model_path: str, engine_path: str = '', device: str = 'cpu') -> tuple:
    """
    Evaluate forces via MetatomicDlopen (dlopen libmetatomic_engine.so). engine_path may be empty to use RGPOT_METATOMIC_ENGINE / package-bundled multi-ABI engine under rgpot/lib/torch-X.Y/. Each call builds its own frontend (F4).
    """

class LJPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None:
        """
        Shifted 12-6 Lennard-Jones. u0 is the well depth (eV), cutoff and psi are Angstrom. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class LJClusterPot:
    def __init__(self, u0: float = 1.0, cutoff: float = 15.0, psi: float = 1.0, switch_width: float = 0.0) -> None:
        """
        Free-boundary shifted 12-6 Lennard-Jones. u0 is the well depth (eV), cutoff and psi are Angstrom. An input cell does not fold pair vectors. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class MorsePot:
    def __init__(self, De: float = 0.7102, a: float = 1.6047, re: float = 2.897, cutoff: float = 9.5, switch_width: float = 0.0) -> None:
        """
        Shifted Morse pair potential. De is the well depth (eV), a is 1/Angstrom, re and cutoff are Angstrom. Defaults are the platinum parameters. A positive switch_width (Angstrom) replaces the shift with the C^2 quintic switch over [cutoff - switch_width, cutoff]; it must not exceed cutoff.
        """

    def config(self) -> dict: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class SWPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class EDIPPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class LenoskyPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class TersoffPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class EAMAlPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class FeHePot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class CuH2Pot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class WaterHPot:
    def __init__(self) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class StillingerWeberSi:
    def __init__(self) -> None:
        """
        Stillinger-Weber silicon, Stillinger and Weber, Phys. Rev. B 31, 5262 (1985). Epsilon is the published 2.16723920 eV, not eOn's 2.3149945 eV (ratio 1.068176741337). Energy is eV, lengths are Angstrom. The gradient is dE/dx.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Return (energy, forces, variance) for one configuration. Forces are -dE/dx.
        """

    def caps(self) -> dict: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and the Cauchy stress, shape (3, 3), in eV/Angstrom^3. With energies=True, also the energy of each atom, shape (n_atoms,), three-body terms on the centre atom. The kernel runs detached.
        """

    @staticmethod
    def params() -> dict:
        """The parameters under the names of a LAMMPS sw file entry."""

    def energy_gradient(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Energy and dE/dx. positions is (n_atoms, 3) or (n_config, n_atoms, 3). A batch shares atom_types and box, and crosses into the library once.
        """

class TersoffSiC:
    def __init__(self) -> None:
        """
        Tersoff silicon-carbon kernel, Tersoff, Phys. Rev. B 39, 5566 (1989). The 1990 erratum sets carbon B to 346.74 eV and d to 4.3484. Carbon beta is dimensionless. Atomic numbers 14 and 6. This class always evaluates the 1989 kernel. The engine name tersoff uses the Fortran silicon gather when every atom is silicon. Energy is eV, lengths are Angstrom. The gradient is dE/dx.
        """

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Return (energy, forces, variance) for one configuration. Forces are -dE/dx.
        """

    def caps(self) -> dict: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and the Cauchy stress, shape (3, 3), in eV/Angstrom^3. With energies=True, also the energy of each atom, shape (n_atoms,), each bond term on its centre atom. The kernel runs detached.
        """

    def params(self) -> dict:
        """
        species maps C and Si to their angular and bond-order parameters. pairs maps (centre, partner) to lambda1, lambda2, A, B (chi included), and the inner and outer cutoffs rin and rout, which are R - D and R + D of a LAMMPS tersoff file.
        """

    def energy_gradient(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """
        Energy and dE/dx. positions is (n_atoms, 3) or (n_config, n_atoms, 3). A batch shares atom_types and box, and crosses into the library once.
        """

class SuttonChenPot:
    def __init__(self, epsilon: float = 1.0, c: float = 34.408, a: float = 1.0, n: int = 10, m: int = 8, cutoff: float = 40.0, periodic: bool = False, stress: bool = False) -> None:
        """
        Sutton-Chen, Phil. Mag. Lett. 61, 139 (1990). epsilon is the energy unit, a and cutoff the length unit, with no energy shift at the cutoff. Defaults are the reduced 10-8 Pt/Au set as a cluster. periodic=True counts every image inside the cutoff. stress=True writes the Cauchy stress.
        """

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class ZBLPot:
    def __init__(self, cut_inner: float = 2.0, cut_global: float = 2.5) -> None:
        """
        Ziegler-Biersack-Littmark screened nuclear repulsion with the LAMMPS pair_zbl switch from cut_inner to cut_global (Angstrom). Free boundaries; writes the Cauchy stress.
        """

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class D3Pot:
    def __init__(self, damping: str = 'bj', functional: str = 'pbe', atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class D4Pot:
    def __init__(self, functional: str = 'pbe', charge: float = 0.0, atm: bool = True) -> None: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
        """

    def evaluate_batch(self, positions: list, atom_types: list, box: list, n_threads: int = 0) -> list:
        """
        One detached BatchEvaluator call. results[i] matches a serial call on inputs[i], bit for bit. n_threads 0 uses the hardware concurrency.
        """

class ExprPot:
    def __init__(self, expression: str, terms: dict) -> None:
        """
        expression is a Lepton string; terms maps names to 'lj'/'morse'/'d3'/'d4' or a dict with kind plus that kind's fields: u0, cutoff, psi, switch_width (lj); De, a, re, cutoff, switch_width (morse); the D3/D4 fields.
        """

    @property
    def expression(self) -> str: ...

    def d_energy_d_term(self, name: str) -> float: ...

    def caps(self) -> dict: ...

    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple: ...

    def evaluate(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], energies: bool = False) -> dict:
        """
        Energy, forces, variance, and stress when the kernel writes it. With energies=True, also the energy of each atom, shape (n_atoms,), when caps()['energies'] is true; ValueError otherwise. F4: the kernel runs detached.
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

class LammpsPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

class GromacsPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

    def native_u0(self) -> float:
        """Well depth stored for GROMACS, kJ/mol."""

    def native_cutoff(self) -> float:
        """Cutoff stored for GROMACS, nanometres."""

    def native_psi(self) -> float:
        """Zero crossing stored for GROMACS, nanometres."""

class OpenMMPot:
    def __call__(self, positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], atom_types: object, box: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)]) -> tuple:
        """Energy, forces and variance. Positions are Angstrom, energy is eV."""

    def native_u0(self) -> float:
        """Well depth stored for OpenMM, kJ/mol."""

    def native_cutoff(self) -> float:
        """Cutoff stored for OpenMM, nanometres."""

    def native_psi(self) -> float:
        """Zero crossing stored for OpenMM, nanometres."""

def rgpot_lammps_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> LammpsPot:
    """Shifted 12-6 Lennard-Jones. The constructor dlopens liblj_engine.so (or liblammps_engine.so) and that plugin stays the backend."""

def rgpot_gromacs_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> GromacsPot:
    """Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once."""

def rgpot_openmm_potential(u0: float = 1.0, cutoff: float = 8.0, psi: float = 1.0, engine_path: str = '') -> OpenMMPot:
    """Shifted 12-6 Lennard-Jones. Nanometres and kJ/mol are stored once."""

class XcKernel:
    def __init__(self, name: str) -> None: ...

    @staticmethod
    def catalog() -> list[str]: ...

    @property
    def name(self) -> str: ...

    def scal_names(self) -> list[str]: ...

    def contract(self, chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object | None = None, scal: dict | None = None, lapl_chi: object | None = None, hess_chi: object | None = None) -> Annotated[NDArray[numpy.float64], dict(order='C')]: ...

    @staticmethod
    def fields_from_density(chi: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], dchi: object, density: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], lapl_chi: object | None = None, hess_chi: object | None = None) -> dict: ...

def neighbor_list(positions: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], cell: Annotated[NDArray[numpy.float64], dict(order='C', device='cpu', writable=False)], cutoff: float, quantities: str = 'ijd', full: bool = False, offsets: bool = False) -> object:
    """
    Neighbour arrays selected by quantities. i and j are indices, d is the distance, D is r_j - r_i with shape (npairs, 3), S is the integer shift with that shape. offsets=True appends a CSR row pointer of length natoms + 1 after sorting pairs by i, then j. `i, j, d = rgpot.neighbor_list(positions, cell, cutoff)`.
    """
