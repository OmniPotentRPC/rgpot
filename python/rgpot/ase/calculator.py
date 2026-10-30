"""In-process ASE calculator over an rgpot potential.

Units are ASE units: energy in eV, forces in eV/Angstrom, stress in
eV/Angstrom^3. Stress is Voigt order ``(xx, yy, zz, yz, xz, xy)`` with
ASE's sign, ``sigma = (1/V) dE/dstrain``.

``free_energy`` equals ``energy``. The classical potentials have no
electronic entropy, so the two results are the same value.

The ASE calculator contract is that of
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
doi:10.1088/1361-648X/aa680e.

``protocol_reentrant`` is True for :meth:`RgpotCalculator.__rg_evaluate__`
only (F5). :meth:`calculate` and ``get_property`` mutate ``self.atoms``
and ``self.results`` and must not run on two threads at once. The
potential object underneath keeps one workspace per calling thread (F2).

The bound kernels are fully periodic. A structure with every periodic
flag false is a cluster: the cell is ignored, and the kernel receives a
cubic box whose side is at least 100 Angstrom and at least the cluster
extent plus twice the kernel cutoff. A mixed periodic flag raises
``ValueError`` and names the non-periodic axes.
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

# Neighbour cutoffs of the bound kernels, in Angstrom. LJ reads
# config()["cutoff"] instead. WaterH has no cutoff, so an open cell for
# it uses the 100 Angstrom floor.
_KERNEL_CUTOFF_ANGSTROM = {
    "SWPot": 1.8 * 2.0951,
    "StillingerWeberSi": 1.8 * 2.0951,
    "EDIPPot": 3.1213820,
    "LenoskyPot": 4.5,
    "TersoffPot": 3.0,
    "EAMAlPot": 7.1,
    "FeHePot": 5.4,
    "CuH2Pot": 6.1,
    "WaterHPot": 0.0,
}


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


def _kernel_cutoff(potential: Any) -> float:
    """Cutoff the open-cell box has to clear, in Angstrom."""
    config = getattr(potential, "config", None)
    if callable(config):
        cfg = config()
        if isinstance(cfg, dict) and "cutoff" in cfg:
            return float(cfg["cutoff"])
    return float(_KERNEL_CUTOFF_ANGSTROM.get(type(potential).__name__, 0.0))


def _pbc_flags(pbc: Any) -> np.ndarray:
    flags = np.asarray(pbc, dtype=bool).reshape(-1)
    if flags.size == 1:
        flags = np.repeat(flags, 3)
    if flags.size != 3:
        raise ValueError(f"pbc must be one flag or three, got {int(flags.size)}")
    return flags


def _cluster_extent(positions: Any) -> float:
    """Longest cartesian span of the cluster, in Angstrom."""
    pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pos.shape[0] == 0:
        return 0.0
    span = np.max(pos, axis=0) - np.min(pos, axis=0)
    return float(np.max(span))


def _open_box(positions: Any, cutoff: float) -> np.ndarray:
    """Cubic cell for a non-periodic cluster.

    The side is at least 100 Angstrom and at least the cluster extent
    plus twice the kernel cutoff. It is also longer than twice the
    extent, so a fully periodic kernel does not fold the cluster.
    """
    extent = _cluster_extent(positions)
    pad = extent + 2.0 * float(cutoff)
    side = max(100.0, pad)
    if side <= 2.0 * extent:
        side = 2.0 * extent + 2.0 * float(cutoff)
    return np.eye(3, dtype=np.float64) * side


def _box(positions: Any, cell: Any, pbc: Any, cutoff: float) -> np.ndarray:
    """Row-major cell passed to a fully periodic kernel."""
    flags = _pbc_flags(pbc)
    if not bool(flags.any()):
        return _open_box(positions, cutoff)
    if not bool(flags.all()):
        names = [
            axis for axis, flag in zip(("x", "y", "z"), flags) if not bool(flag)
        ]
        joined = ", ".join(names)
        raise ValueError(
            f"kernels are fully periodic; non-periodic axes: {joined}"
        )
    return np.ascontiguousarray(np.asarray(cell, dtype=np.float64).reshape(3, 3))


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
    # The pure protocol may run concurrently. calculate() writes
    # self.atoms and self.results, so this name is not "calculate is
    # reentrant".
    protocol_reentrant = True

    def __init__(self, potential: Any, **kwargs: Any) -> None:
        pot_kw = {k: v for k, v in kwargs.items() if k not in _ASE_KEYS}
        ase_kw = {k: v for k, v in kwargs.items() if k in _ASE_KEYS}
        self._potential = _resolve_potential(potential, pot_kw)
        self._cutoff = _kernel_cutoff(self._potential)
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
        box = _box(pos, cell, pbc, self._cutoff)
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
            boxes.append(
                _box(positions[-1], atoms.cell.array, atoms.pbc, self._cutoff)
            )
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
