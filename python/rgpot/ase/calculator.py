"""In-process ASE calculator over an rgpot potential.

Units are ASE units: energy in eV, forces in eV/Angstrom, stress in
eV/Angstrom^3. Stress is Voigt order ``(xx, yy, zz, yz, xz, xy)`` with
ASE's sign, ``sigma = (1/V) dE/dstrain``.

``free_energy`` equals ``energy``. The classical potentials have no
electronic entropy, so the two results are the same value.

The ASE calculator contract is that of
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
doi:10.1088/1361-648X/aa680e.

``reentrant`` is True for :meth:`RgpotCalculator.__rg_evaluate__` only
(F5). :meth:`calculate` and ``get_property`` mutate ``self.atoms`` and
``self.results`` and must not run on two threads at once. The potential
object underneath is reentrant: one workspace per calling thread (F2).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator

_ASE_KEYS = {
    "restart",
    "ignore_bad_restart_file",
    "label",
    "atoms",
    "directory",
}

# kind string -> (class name on rgpot, accepts LJ-style numeric kwargs)
_KINDS: dict[str, str] = {
    "lj": "LJPot",
    "lennard-jones": "LJPot",
    "lennardjones": "LJPot",
    "eam_al": "EAMAlPot",
    "eam-al": "EAMAlPot",
    "sw": "SWPot",
    "edip": "EDIPPot",
    "lenosky": "LenoskyPot",
    "tersoff": "TersoffPot",
    "fehe": "FeHePot",
    "cuh2": "CuH2Pot",
    "water_h": "WaterHPot",
    "d3": "D3Pot",
    "d4": "D4Pot",
}

_LJ_KEYS = {"u0", "cutoff", "psi", "switch_width"}
_D3_KEYS = {"damping", "functional", "atm"}
_D4_KEYS = {"functional", "charge", "atm"}


def _resolve_potential(potential: Any, params: dict[str, Any]) -> Any:
    import rgpot

    if not isinstance(potential, str):
        if params:
            raise ValueError(
                "potential parameters are accepted only with a kind string"
            )
        if not hasattr(potential, "evaluate"):
            raise TypeError(
                "potential must be an rgpot potential or a kind string"
            )
        return potential
    key = potential.strip().lower().replace(" ", "_")
    class_name = _KINDS.get(key)
    if class_name is None:
        raise ValueError(f"unknown potential kind {potential!r}")
    cls = getattr(rgpot, class_name, None)
    if cls is None:
        raise ValueError(f"potential {potential!r} is not in this build")
    if class_name == "LJPot":
        unknown = set(params) - _LJ_KEYS
        if unknown:
            raise ValueError(f"unknown LJ parameters: {sorted(unknown)}")
        return cls(
            u0=float(params.get("u0", 1.0)),
            cutoff=float(params.get("cutoff", 15.0)),
            psi=float(params.get("psi", 1.0)),
            switch_width=float(params.get("switch_width", 0.0)),
        )
    if class_name == "D3Pot":
        unknown = set(params) - _D3_KEYS
        if unknown:
            raise ValueError(f"unknown D3 parameters: {sorted(unknown)}")
        return cls(
            damping=str(params.get("damping", "bj")),
            functional=str(params.get("functional", "pbe")),
            atm=bool(params.get("atm", True)),
        )
    if class_name == "D4Pot":
        unknown = set(params) - _D4_KEYS
        if unknown:
            raise ValueError(f"unknown D4 parameters: {sorted(unknown)}")
        return cls(
            functional=str(params.get("functional", "pbe")),
            charge=float(params.get("charge", 0.0)),
            atm=bool(params.get("atm", True)),
        )
    if params:
        raise ValueError(f"{class_name} does not take parameters")
    return cls()


def _box(cell: Any, pbc: Any) -> np.ndarray:
    """Row-major cell. A zero cell with no periodic axes becomes 100 Angstrom.

    The determinant is three multiplies. ``numpy.linalg.det`` on every
    frame of a batch costs more than an EAM evaluation of Pt7.
    """
    box = np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(3, 3))
    flags = np.asarray(pbc, dtype=bool).reshape(-1)
    if flags.any():
        return box
    a = box[0]
    b = box[1]
    c = box[2]
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(float(det)) < 1e-18:
        return np.eye(3, dtype=np.float64) * 100.0
    return box


def _voigt(stress: np.ndarray) -> np.ndarray:
    """Row-major 3x3 Cauchy stress to ASE Voigt (xx, yy, zz, yz, xz, xy)."""
    s = np.asarray(stress, dtype=np.float64).reshape(3, 3)
    return np.ascontiguousarray(
        [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]],
        dtype=np.float64,
    )


def _as_f64(array: Any, what: str) -> np.ndarray:
    out = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    if out.dtype != np.float64:
        raise TypeError(f"{what} must be float64")
    return out


class RgpotCalculator(Calculator):
    """ASE calculator that evaluates an rgpot potential in process.

    Parameters
    ----------
    potential:
        An rgpot potential object, or a kind string such as ``"lj"`` or
        ``"eam_al"``. Keyword arguments that are not ASE calculator
        arguments are potential parameters (``u0``, ``cutoff``, ``psi``,
        ``switch_width`` for LJ).
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    # F5: this flag is the pure protocol, not calculate(). calculate()
    # mutates self.atoms and self.results and is not reentrant.
    reentrant = True

    def __init__(self, potential: Any, **kwargs: Any) -> None:
        pot_kw = {k: v for k, v in kwargs.items() if k not in _ASE_KEYS}
        ase_kw = {k: v for k, v in kwargs.items() if k in _ASE_KEYS}
        self._potential = _resolve_potential(potential, pot_kw)
        Calculator.__init__(self, **ase_kw)
        caps = dict(self._potential.caps())
        props = ["energy", "free_energy", "forces"]
        if bool(caps.get("stress")):
            props.append("stress")
        if bool(caps.get("energies")):
            props.append("energies")
        self.implemented_properties = props
        self._caps = caps

    @property
    def potential(self) -> Any:
        """The rgpot potential this calculator evaluates."""
        return self._potential

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: Sequence[str] = ("energy",),
        system_changes: Any = all_changes,
    ) -> None:
        """Fill ``self.results``. ``free_energy`` equals ``energy``."""
        Calculator.calculate(self, atoms, properties, system_changes)
        assert self.atoms is not None
        evaluated = self.__rg_evaluate__(
            self.atoms.get_positions(),
            self.atoms.get_atomic_numbers(),
            np.asarray(self.atoms.cell.array),
            self.atoms.pbc,
        )
        self.results["energy"] = evaluated["energy"]
        self.results["free_energy"] = evaluated["energy"]
        self.results["forces"] = evaluated["forces"]
        if "stress" in self.implemented_properties and "stress" in evaluated:
            self.results["stress"] = evaluated["stress"]
        if "energies" in self.implemented_properties and "energies" in evaluated:
            self.results["energies"] = evaluated["energies"]

    def __rg_evaluate__(
        self,
        positions: Any,
        numbers: Any,
        cell: Any,
        pbc: Any,
    ) -> dict[str, Any]:
        """Pure evaluation. Does not read or write ``self`` state (F5).

        Returns energy (eV), free_energy (equal to energy), forces
        (eV/Angstrom, shape (N, 3)), and stress in ASE Voigt order when
        the potential reports it.
        """
        pos = _as_f64(positions, "positions")
        num = np.ascontiguousarray(np.asarray(numbers, dtype=np.int32))
        box = _box(cell, pbc)
        raw = self._potential.evaluate(pos, num, box)
        energy = float(raw["energy"])
        forces = np.ascontiguousarray(raw["forces"], dtype=np.float64)
        out: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
            "forces": forces,
        }
        stress = raw.get("stress")
        if stress is not None:
            out["stress"] = _voigt(np.asarray(stress))
        energies = raw.get("energies")
        if energies is not None:
            out["energies"] = np.ascontiguousarray(energies, dtype=np.float64)
        return out

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Analytic Hessian, shape (3N, 3N).

        Raises NotImplementedError. No bound potential reports an analytic
        Hessian.
        """
        del atoms
        raise NotImplementedError(
            f"{type(self._potential).__name__} has no analytic Hessian"
        )

    def calculate_many(
        self,
        frames: Iterable[Atoms],
        properties: Sequence[str] = ("energy", "forces"),
        *,
        attach: bool = False,
        threads: int = 0,
    ) -> list[dict[str, Any]]:
        """Evaluate many structures in one detached batch.

        Results match N single :meth:`calculate` calls, bit for bit, for
        energy and forces. Inputs are not mutated. ``attach=True`` writes
        a :class:`~ase.calculators.singlepoint.SinglePointCalculator` onto
        each :class:`~ase.Atoms`.

        ``threads`` is the BatchEvaluator worker count. 0 selects the
        hardware concurrency. The worker count is not part of the result.
        """
        wanted = tuple(properties)
        frames_list = list(frames)
        positions: list[np.ndarray] = []
        numbers: list[np.ndarray] = []
        boxes: list[np.ndarray] = []
        for atoms in frames_list:
            # Read the arrays ASE stores. get_positions copies, and that
            # copy was comparable to an EAM evaluation of Pt7. The C++
            # batch copies into BatchInput before it detaches, so the
            # Atoms objects are not mutated and are not read again.
            positions.append(_as_f64(atoms.positions, "positions"))
            numbers.append(np.ascontiguousarray(atoms.numbers, dtype=np.int32))
            boxes.append(_box(atoms.cell.array, atoms.pbc))
        raw_rows = self._potential.evaluate_batch(
            positions, numbers, boxes, int(threads)
        )
        rows: list[dict[str, Any]] = []
        for index, (atoms, raw) in enumerate(zip(frames_list, raw_rows)):
            energy = float(raw["energy"])
            forces = np.ascontiguousarray(raw["forces"], dtype=np.float64)
            row: dict[str, Any] = {}
            if "energy" in wanted or "free_energy" in wanted:
                row["energy"] = energy
                row["free_energy"] = energy
            if "forces" in wanted:
                row["forces"] = forces
            if "stress" in wanted and "stress" in self.implemented_properties:
                # BatchEvaluator returns energy and forces. Stress comes
                # from the same kernel on the copied geometry.
                stress_eval = self._potential.evaluate(
                    positions[index], numbers[index], boxes[index]
                )
                if stress_eval.get("stress") is not None:
                    row["stress"] = _voigt(np.asarray(stress_eval["stress"]))
            rows.append(row)
            if attach:
                spc_kw: dict[str, Any] = {}
                if "energy" in row:
                    spc_kw["energy"] = row["energy"]
                if "forces" in row:
                    spc_kw["forces"] = row["forces"]
                if "stress" in row:
                    spc_kw["stress"] = row["stress"]
                atoms.calc = SinglePointCalculator(atoms, **spc_kw)
        return rows
