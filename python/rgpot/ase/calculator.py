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
flag false is a cluster: the kernel receives a cubic box whose side is
at least 100 Angstrom and at least the cluster extent plus twice the
kernel cutoff, so a periodic image cannot enter the cutoff. Energy and
forces do not depend on the caller's cell. Stress does: it is the
cluster virial divided by the caller's cell volume, which is what
``calculate_numerical_stress`` differentiates. A caller with no
three-dimensional cell gets no stress, and ``get_stress`` raises
``PropertyNotImplementedError``. A mixed periodic flag raises
``ValueError`` and names the non-periodic axes. A potential whose caps
report ``periodic`` false accepts only ``pbc=False``.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
)
from ase.calculators.singlepoint import SinglePointCalculator

_ASE_KEYS = {
    "restart",
    "ignore_bad_restart_file",
    "label",
    "atoms",
    "directory",
}

# kind string -> name of the class or factory function on rgpot
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

_LJ_DEFAULTS = {"u0": 1.0, "cutoff": 15.0, "psi": 1.0, "switch_width": 0.0}

# Factory name -> (label used in error messages, default parameters).
# A factory missing here takes no parameters.
_FACTORY_PARAMETERS: dict[str, tuple[str, dict[str, Any]]] = {
    "LJPot": ("LJ", _LJ_DEFAULTS),
    "D3Pot": ("D3", {"damping": "bj", "functional": "pbe", "atm": True}),
    "D4Pot": ("D4", {"functional": "pbe", "charge": 0.0, "atm": True}),
}

# Classes whose objects carry no parameters, so type(obj)() rebuilds them.
_PARAMETERLESS = {
    "SWPot",
    "EDIPPot",
    "LenoskyPot",
    "TersoffPot",
    "EAMAlPot",
    "FeHePot",
    "CuH2Pot",
    "WaterHPot",
}

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


def _check_parameters(factory: str, params: dict[str, Any]) -> None:
    label, defaults = _FACTORY_PARAMETERS.get(factory, (factory, {}))
    if not defaults:
        if params:
            raise ValueError(f"{factory} does not take parameters")
        return
    unknown = set(params) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {label} parameters: {sorted(unknown)}")


def _coerce(default: Any, value: Any) -> Any:
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def _build(factory: str, params: dict[str, Any]) -> Any:
    """Construct the potential ``factory`` names from complete parameters."""
    import rgpot

    make = getattr(rgpot, factory, None)
    if make is None:
        raise ValueError(f"potential {factory!r} is not in this build")
    _check_parameters(factory, params)
    _, defaults = _FACTORY_PARAMETERS.get(factory, (factory, {}))
    kwargs = {k: _coerce(d, params.get(k, d)) for k, d in defaults.items()}
    return make(**kwargs)


def _resolve_potential(
    potential: Any, params: dict[str, Any]
) -> tuple[Any, str | None, dict[str, Any], dict[str, Any]]:
    """Return (potential, factory, default parameters, parameters).

    ``factory`` is the rgpot name that rebuilds the potential from the
    parameters, or None when the object cannot be rebuilt.
    """
    if isinstance(potential, str):
        key = potential.strip().lower().replace(" ", "_")
        factory = _KINDS.get(key)
        if factory is None:
            raise ValueError(f"unknown potential kind {potential!r}")
        _check_parameters(factory, params)
        defaults = dict(_FACTORY_PARAMETERS.get(factory, (factory, {}))[1])
        full = {**defaults, **params}
        return _build(factory, full), factory, defaults, full
    if not hasattr(potential, "evaluate"):
        raise TypeError("potential must be an rgpot potential or a kind string")
    name = type(potential).__name__
    if name in _PARAMETERLESS:
        if params:
            raise ValueError(f"{name} does not take parameters")
        return potential, name, {}, {}
    config = getattr(potential, "config", None)
    if name in _FACTORY_PARAMETERS and callable(config):
        _check_parameters(name, params)
        current = dict(config())
        defaults = dict(_FACTORY_PARAMETERS[name][1])
        full = {**current, **params}
        built = _build(name, full) if params else potential
        return built, name, defaults, full
    if params:
        raise ValueError(
            "potential parameters are accepted only with a kind string "
            "or a potential that reports config()"
        )
    return potential, None, {}, {}


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


def _cell_volume(cell: Any) -> float:
    """Absolute volume of a 3x3 cell. A singular cell has volume 0."""
    matrix = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    return float(abs(np.linalg.det(matrix)))


def _voigt(stress: np.ndarray) -> np.ndarray:
    """Row-major 3x3 Cauchy stress to ASE Voigt (xx, yy, zz, yz, xz, xy)."""
    s = np.asarray(stress, dtype=np.float64).reshape(3, 3)
    return np.ascontiguousarray(
        [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]],
        dtype=np.float64,
    )


def _ase_stress(
    kernel_stress: Any, kernel_box: Any, user_cell: Any, pbc: Any
) -> np.ndarray | None:
    """Cauchy stress on the cell ASE differentiates, or None.

    A periodic call uses the caller's cell as the kernel box, so the
    kernel stress is already ASE's. A cluster is evaluated in a padded
    cube that keeps periodic images out of the cutoff. The kernel then
    divides the virial by that cube. ASE's finite-difference stress
    divides the same virial by the caller's cell, so the cube stress is
    rescaled by ``V_kernel / V_caller``. A caller with no
    three-dimensional cell has no volume to strain, so there is no
    stress, as for ASE's own calculators.
    """
    user_volume = _cell_volume(user_cell)
    if not user_volume > 0.0:
        return None
    stress = np.asarray(kernel_stress, dtype=np.float64).reshape(3, 3).copy()
    if not bool(_pbc_flags(pbc).any()):
        kernel_volume = _cell_volume(kernel_box)
        if kernel_volume > 0.0:
            stress *= kernel_volume / user_volume
    return _voigt(stress)


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
        ``switch_width`` for LJ). They live in ``parameters``, and
        :meth:`set` rebuilds the potential when one changes.
    """

    implemented_properties = ["energy", "free_energy", "forces"]
    # The pure protocol may run concurrently. calculate() writes
    # self.atoms and self.results, so this name is not "calculate is
    # reentrant".
    protocol_reentrant = True

    def __init__(self, potential: Any, **kwargs: Any) -> None:
        pot_kw = {k: v for k, v in kwargs.items() if k not in _ASE_KEYS}
        ase_kw = {k: v for k, v in kwargs.items() if k in _ASE_KEYS}
        built, factory, defaults, params = _resolve_potential(potential, pot_kw)
        self._factory = factory
        self.default_parameters = defaults
        self._install(built)
        Calculator.__init__(self, **ase_kw)
        self.parameters.update(params)

    def _install(self, potential: Any) -> None:
        self._potential = potential
        self._cutoff = _kernel_cutoff(potential)
        caps = dict(potential.caps())
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

    def set(self, **kwargs: Any) -> dict[str, Any]:
        """Change potential parameters. Returns the keys that changed.

        A change rebuilds the potential and clears ``results``. An
        unknown key raises ``ValueError`` before anything is stored.
        """
        if "parameters" in kwargs or not kwargs:
            return Calculator.set(self, **kwargs)
        if self._factory is None:
            name = type(self._potential).__name__
            raise ValueError(
                f"{name} was given as an object without config(); "
                "build the calculator from a kind string to set parameters"
            )
        _check_parameters(self._factory, kwargs)
        new = {**self.parameters, **kwargs}
        changed = Calculator.set(self, **kwargs)
        if changed:
            self._install(_build(self._factory, new))
            self.reset()
        return changed

    def __getstate__(self) -> dict[str, Any]:
        if self._factory is None:
            name = type(self._potential).__name__
            raise TypeError(
                f"cannot copy or pickle RgpotCalculator over a {name} object "
                "without config(); build it from a kind string"
            )
        state = self.__dict__.copy()
        state.pop("_potential", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._install(_build(self._factory, dict(self.parameters)))

    def _check_periodic(self, pbc: Any) -> None:
        if bool(_pbc_flags(pbc).any()) and self._caps.get("periodic") is False:
            name = type(self._potential).__name__
            raise ValueError(
                f"{name} has free boundaries; set pbc=False on the Atoms"
            )

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
        user_cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self._check_periodic(pbc)
        box = _box(pos, user_cell, pbc, self._cutoff)
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
            voigt = _ase_stress(stress, box, user_cell, pbc)
            if voigt is not None:
                out["stress"] = voigt
        energies = raw.get("energies")
        if energies is not None:
            out["energies"] = np.ascontiguousarray(energies, dtype=np.float64)
        return out

    def get_hessian(self, atoms: Atoms) -> np.ndarray:
        """Analytic Hessian, shape (3N, 3N).

        Raises PropertyNotImplementedError, a NotImplementedError. No
        bound potential reports an analytic Hessian.
        """
        del atoms
        raise PropertyNotImplementedError(
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
        user_cells: list[np.ndarray] = []
        pbcs: list[Any] = []
        for atoms in frames_list:
            # Read the arrays ASE stores. get_positions copies, and that
            # copy was comparable to an EAM evaluation of Pt7. The C++
            # batch copies into BatchInput before it detaches, so the
            # Atoms objects are not mutated and are not read again.
            positions.append(_as_f64(atoms.positions, "positions"))
            numbers.append(np.ascontiguousarray(atoms.numbers, dtype=np.int32))
            user_cells.append(np.asarray(atoms.cell.array, dtype=np.float64))
            pbcs.append(atoms.pbc)
            self._check_periodic(atoms.pbc)
            boxes.append(
                _box(positions[-1], user_cells[-1], atoms.pbc, self._cutoff)
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
                    voigt = _ase_stress(
                        stress_eval["stress"],
                        boxes[index],
                        user_cells[index],
                        pbcs[index],
                    )
                    if voigt is not None:
                        row["stress"] = voigt
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
