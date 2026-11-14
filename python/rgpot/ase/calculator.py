"""In-process ASE calculator over an rgpot potential.

Units are ASE units: energy in eV, forces in eV/Angstrom, stress in
eV/Angstrom^3. Stress is Voigt order ``(xx, yy, zz, yz, xz, xy)`` with
ASE's sign, ``sigma = (1/V) dE/dstrain``.

``free_energy`` equals ``energy``. The classical potentials have no
electronic entropy, so the two results are the same value.

``energies`` (``get_potential_energies``) is implemented only for a
potential whose ``caps()["energies"]`` is true. Pair terms are split half
and half, an embedding term sits on the embedded atom and a three-body
term on its centre atom, and the shares sum to ``energy``. They are
computed only for a call that asks for them.

The ASE calculator contract is that of
Hjorth Larsen et al., J. Phys.: Condens. Matter 29, 273002 (2017),
doi:10.1088/1361-648X/aa680e.

``protocol_reentrant`` is True for :meth:`RgpotCalculator.__rg_evaluate__`
only (F5). :meth:`calculate` and ``get_property`` mutate ``self.atoms``
and ``self.results`` and must not run on two threads at once. The
potential object underneath keeps one workspace per calling thread (F2).

The bound kernels are fully periodic and sum every periodic image inside
the cutoff, so a cell thinner than twice the cutoff, a one-atom
primitive cell included, is evaluated as it is. A structure with every
periodic flag false is a cluster: the kernel receives a cubic box whose side is
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
    "lj_cluster": "LJClusterPot",
    "lj-cluster": "LJClusterPot",
    "ljcluster": "LJClusterPot",
    "morse": "MorsePot",
    "eam_al": "EAMAlPot",
    "eam-al": "EAMAlPot",
    "sw": "SWPot",
    "stillinger_weber_si": "StillingerWeberSi",
    "sw_si": "StillingerWeberSi",
    "edip": "EDIPPot",
    "lenosky": "LenoskyPot",
    "tersoff": "TersoffPot",
    "tersoff_sic": "TersoffSiC",
    "fehe": "FeHePot",
    "cuh2": "CuH2Pot",
    "water_h": "WaterHPot",
    "d3": "D3Pot",
    "d4": "D4Pot",
    "expr": "ExprPot",
    "lammps": "rgpot_lammps_potential",
    "gromacs": "rgpot_gromacs_potential",
    "openmm": "rgpot_openmm_potential",
}

_LJ_DEFAULTS = {"u0": 1.0, "cutoff": 15.0, "psi": 1.0, "switch_width": 0.0}
_MD_LJ_DEFAULTS = {"u0": 1.0, "cutoff": 8.0, "psi": 1.0, "engine_path": ""}

# Factory name -> (label used in error messages, default parameters).
# A factory missing here takes no parameters. A None default is required.
_FACTORY_PARAMETERS: dict[str, tuple[str, dict[str, Any]]] = {
    "LJPot": ("LJ", _LJ_DEFAULTS),
    "LJClusterPot": ("LJ", _LJ_DEFAULTS),
    "MorsePot": (
        "Morse",
        {"De": 0.7102, "a": 1.6047, "re": 2.8970, "cutoff": 9.5, "switch_width": 0.0},
    ),
    "D3Pot": ("D3", {"damping": "bj", "functional": "pbe", "atm": True}),
    "D4Pot": ("D4", {"functional": "pbe", "charge": 0.0, "atm": True}),
    "ExprPot": ("ExprPot", {"expression": None, "terms": None}),
    "rgpot_lammps_potential": ("LAMMPS LJ", _MD_LJ_DEFAULTS),
    "rgpot_gromacs_potential": ("GROMACS LJ", _MD_LJ_DEFAULTS),
    "rgpot_openmm_potential": ("OpenMM LJ", _MD_LJ_DEFAULTS),
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
    "StillingerWeberSi",
    "TersoffSiC",
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
    "TersoffSiC": 3.0,
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
    if factory == "ExprPot":
        missing = [k for k in ("expression", "terms") if params.get(k) is None]
        if missing:
            raise ValueError(f"ExprPot needs {' and '.join(missing)}")
        return make(str(params["expression"]), dict(params["terms"]))
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
    if not (hasattr(potential, "evaluate") or callable(potential)):
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


class _CallAdapter:
    """``evaluate`` for a potential that has only ``__call__``.

    ``__call__`` returns (energy, forces, variance). A potential with a
    ``stress()`` method (IPIPot) reports the Cauchy stress of its last
    call there, or None when the engine sent none.
    """

    def __init__(self, potential: Any) -> None:
        self.potential = potential
        self._stress = getattr(potential, "stress", None)

    def caps(self) -> dict[str, Any]:
        return {"periodic": True, "stress": callable(self._stress)}

    def evaluate(self, positions: Any, numbers: Any, box: Any) -> dict[str, Any]:
        energy, forces, _variance = self.potential(positions, numbers, box)
        stress = self._stress() if callable(self._stress) else None
        return {"energy": float(energy), "forces": forces, "stress": stress}

    def evaluate_batch(
        self, positions: Any, numbers: Any, boxes: Any, threads: int = 0
    ) -> list[dict[str, Any]]:
        del threads
        return [
            self.evaluate(p, n, b) for p, n, b in zip(positions, numbers, boxes)
        ]


def _kernel_of(potential: Any) -> Any:
    if hasattr(potential, "evaluate"):
        return potential
    return _CallAdapter(potential)


def _caps_of(kernel: Any) -> dict[str, Any]:
    caps = getattr(kernel, "caps", None)
    return dict(caps()) if callable(caps) else {}


def _expr_term_cutoff(term: Any) -> float:
    """Cutoff of one ExprPot child spec. D3 and D4 children are free-boundary."""
    if isinstance(term, dict):
        if "cutoff" in term:
            return float(term["cutoff"])
        kind = term.get("kind", term.get("type"))
    else:
        kind = term
    factory = {"lj": "LJPot", "morse": "MorsePot"}.get(str(kind))
    if factory is None:
        return 0.0
    return float(_FACTORY_PARAMETERS[factory][1]["cutoff"])


def _kernel_cutoff(potential: Any, params: dict[str, Any] | None = None) -> float:
    """Interaction range of the kernel in Angstrom, 0 when unknown."""
    config = getattr(potential, "config", None)
    if callable(config):
        cfg = config()
        if isinstance(cfg, dict) and "cutoff" in cfg:
            return float(cfg["cutoff"])
    params = params or {}
    cutoffs = [
        _expr_term_cutoff(term) for term in dict(params.get("terms") or {}).values()
    ]
    if "cutoff" in params:
        cutoffs.append(float(params["cutoff"]))
    if cutoffs:
        return max(cutoffs)
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
        self._install(built, params)
        Calculator.__init__(self, **ase_kw)
        self.parameters.update(params)

    def _install(self, potential: Any, params: dict[str, Any]) -> None:
        self._potential = potential
        self._kernel = _kernel_of(potential)
        self._cutoff = _kernel_cutoff(potential, params)
        caps = _caps_of(self._kernel)
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
            self._install(_build(self._factory, new), new)
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
        for key in ("_potential", "_kernel"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        params = dict(self.parameters)
        self._install(_build(self._factory, params), params)

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
            energies="energies" in properties
            and "energies" in self.implemented_properties,
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
        energies: bool = False,
    ) -> dict[str, Any]:
        """Pure evaluation. Does not read or write ``self`` state (F5).

        Returns energy (eV), free_energy (equal to energy), forces
        (eV/Angstrom, shape (N, 3)), and stress in ASE Voigt order when
        the potential reports it. ``energies=True`` adds the energy of
        each atom (eV, shape (N,)), which sums to ``energy``. The kernel
        computes them only when asked. A potential without them raises
        ``PropertyNotImplementedError``.
        """
        pos = _as_f64(positions, "positions")
        num = np.ascontiguousarray(np.asarray(numbers, dtype=np.int32))
        user_cell = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        kernel_in = self._kernel_input(pos, num, user_cell, pbc)
        raw = self._evaluate(kernel_in, energies)
        return self._from_kernel(raw, len(num), kernel_in[2], user_cell, pbc)

    def _evaluate(
        self, kernel_in: tuple[np.ndarray, np.ndarray, np.ndarray], energies: bool
    ) -> dict[str, Any]:
        if not energies:
            return self._kernel.evaluate(*kernel_in)
        if not bool(self._caps.get("energies")):
            raise PropertyNotImplementedError(
                f"{type(self._potential).__name__} has no per-atom energies"
            )
        return self._kernel.evaluate(*kernel_in, energies=True)

    def _kernel_input(
        self, pos: np.ndarray, num: np.ndarray, user_cell: np.ndarray, pbc: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions, numbers and box the kernel sees for one structure."""
        self._check_periodic(pbc)
        return pos, num, _box(pos, user_cell, pbc, self._cutoff)

    @staticmethod
    def _from_kernel(
        raw: dict[str, Any],
        n_atoms: int,
        box: np.ndarray,
        user_cell: np.ndarray,
        pbc: Any,
    ) -> dict[str, Any]:
        """Results for the caller's atoms from a kernel result on ``box``."""
        forces = np.asarray(raw["forces"], dtype=np.float64).reshape(n_atoms, 3)
        energy = float(raw["energy"])
        out: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.ascontiguousarray(forces),
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
            pos, num, box = self._kernel_input(
                _as_f64(atoms.positions, "positions"),
                np.ascontiguousarray(atoms.numbers, dtype=np.int32),
                np.asarray(atoms.cell.array, dtype=np.float64),
                atoms.pbc,
            )
            positions.append(pos)
            numbers.append(num)
            boxes.append(box)
            user_cells.append(np.asarray(atoms.cell.array, dtype=np.float64))
            pbcs.append(atoms.pbc)
        batch = getattr(self._kernel, "evaluate_batch", None)
        if batch is not None:
            raw_rows = batch(positions, numbers, boxes, int(threads))
        else:
            raw_rows = [
                self._kernel.evaluate(p, n, b)
                for p, n, b in zip(positions, numbers, boxes)
            ]
        rows: list[dict[str, Any]] = []
        props = self.implemented_properties
        want_stress = "stress" in wanted and "stress" in props
        want_energies = "energies" in wanted and "energies" in props
        for index, (atoms, raw) in enumerate(zip(frames_list, raw_rows)):
            if want_stress or want_energies:
                # BatchEvaluator returns energy and forces. Stress and
                # per-atom energies come from the same kernel on the
                # copied geometry.
                extra = self._evaluate(
                    (positions[index], numbers[index], boxes[index]), want_energies
                )
                raw = {
                    **raw,
                    "stress": extra.get("stress"),
                    "energies": extra.get("energies"),
                }
            result = self._from_kernel(
                raw, len(atoms), boxes[index], user_cells[index], pbcs[index]
            )
            row: dict[str, Any] = {}
            if "energy" in wanted or "free_energy" in wanted:
                row["energy"] = result["energy"]
                row["free_energy"] = result["free_energy"]
            if "forces" in wanted:
                row["forces"] = result["forces"]
            if "stress" in wanted and "stress" in result:
                row["stress"] = result["stress"]
            if "energies" in wanted and "energies" in result:
                row["energies"] = result["energies"]
            rows.append(row)
            if attach:
                spc_kw: dict[str, Any] = {}
                if "energy" in row:
                    spc_kw["energy"] = row["energy"]
                if "forces" in row:
                    spc_kw["forces"] = row["forces"]
                if "stress" in row:
                    spc_kw["stress"] = row["stress"]
                if "energies" in row:
                    spc_kw["energies"] = row["energies"]
                atoms.calc = SinglePointCalculator(atoms, **spc_kw)
        return rows
