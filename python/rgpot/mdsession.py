"""Engine-generic MD sessions over the rgpot MD session C ABI.

A thin ctypes binding of ``rgpot/MDSession/md_session_c_abi.h``. Any engine
library that exports ``<prefix>_md_*`` (``rgpot_nblib`` today) is driven the
same way::

    engine = MDEngine("librgpot_nblib.so", "rgpot_nblib")
    with engine.create(topology_text, time_step_ps=0.002) as md:
        md.set_box(box)
        md.set_positions(x)
        md.set_velocities(v)
        md.step_nve(100)
        state = md.state()

Units are Angstrom, picosecond, eV and amu. The module needs only ctypes and
numpy, so it does not import the compiled ``rgpot`` extension.
"""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

ABI_VERSION = 1

FEATURE_NVE = 1
FEATURE_PROJECTED = 2
FEATURE_HOOK = 4
FEATURE_BOUNDARY = 8
FEATURE_VIRIAL = 16

PHASE_POSITIONS = 1
PHASE_FORCES = 2

_DOUBLE_P = C.POINTER(C.c_double)
_ERR_LEN = 512


class _Config(C.Structure):
    _fields_ = [("bytes", C.c_uint32), ("time_step_ps", C.c_double)]


class _Info(C.Structure):
    _fields_ = [
        ("bytes", C.c_uint32),
        ("potential_ev", C.c_double),
        ("kinetic_ev", C.c_double),
        ("time_ps", C.c_double),
        ("steps", C.c_uint64),
        ("force_evaluations", C.c_uint64),
        ("virial_ev", C.c_double * 9),
        ("box", C.c_double * 9),
    ]


class _Thermostat(C.Structure):
    _fields_ = [
        ("bytes", C.c_uint32),
        ("kt_ev", C.c_double),
        ("friction_per_ps", C.c_double),
    ]


class _Plane(C.Structure):
    _fields_ = [
        ("bytes", C.c_uint32),
        ("dimension", C.c_size_t),
        ("point", _DOUBLE_P),
        ("normal", _DOUBLE_P),
    ]


class _Wall(C.Structure):
    _fields_ = [("normal", _DOUBLE_P), ("offset", C.c_double)]


_HOOK = C.CFUNCTYPE(
    C.c_int,
    C.c_void_p,
    C.c_int,
    C.c_uint64,
    C.c_size_t,
    _DOUBLE_P,
    _DOUBLE_P,
    _DOUBLE_P,
    _DOUBLE_P,
)
_BOUNDARY = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_size_t, _DOUBLE_P, _DOUBLE_P)

# name: (restype, argtypes) of every call after the prefix and "_md_".
_CALLS = {
    "abi_version": (C.c_int, []),
    "abi_minor": (C.c_int, []),
    "available": (C.c_int, []),
    "features": (C.c_uint, []),
    "engine": (C.c_char_p, []),
    "integrator": (C.c_char_p, []),
    "last_error": (C.c_char_p, []),
    "create": (
        C.c_void_p,
        [C.c_char_p, C.POINTER(_Config), C.c_char_p, C.c_size_t],
    ),
    "destroy": (None, [C.c_void_p]),
    "atom_count": (C.c_size_t, [C.c_void_p]),
    "masses": (C.c_int, [C.c_void_p, _DOUBLE_P, C.c_size_t]),
    "set_box": (C.c_int, [C.c_void_p, _DOUBLE_P]),
    "set_positions": (C.c_int, [C.c_void_p, _DOUBLE_P, C.c_size_t]),
    "set_velocities": (C.c_int, [C.c_void_p, _DOUBLE_P, C.c_size_t]),
    "set_time_step": (C.c_int, [C.c_void_p, C.c_double]),
    "set_seed": (C.c_int, [C.c_void_p, C.c_uint64]),
    "initialize": (C.c_int, [C.c_void_p]),
    "state": (
        C.c_int,
        [C.c_void_p, _DOUBLE_P, _DOUBLE_P, _DOUBLE_P, C.c_size_t, C.POINTER(_Info)],
    ),
    "step_nve": (C.c_int, [C.c_void_p, C.c_uint64]),
    "step_projected": (
        C.c_int,
        [
            C.c_void_p,
            C.c_uint64,
            C.POINTER(_Thermostat),
            C.POINTER(_Plane),
            C.POINTER(_Wall),
            C.c_size_t,
        ],
    ),
    "set_hook": (C.c_int, [C.c_void_p, _HOOK, C.c_void_p]),
    "set_boundary": (C.c_int, [C.c_void_p, _BOUNDARY, C.c_void_p]),
}
_REQUIRED = (
    "abi_version",
    "available",
    "features",
    "last_error",
    "create",
    "destroy",
    "atom_count",
    "masses",
    "set_box",
    "set_positions",
    "set_velocities",
    "state",
    "step_nve",
)


class MDError(RuntimeError):
    """A session call failed; the message is the engine's reason."""


@dataclass(frozen=True)
class MDState:
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    potential_ev: float
    kinetic_ev: float
    time_ps: float
    steps: int
    force_evaluations: int
    virial_ev: np.ndarray
    box: np.ndarray


def _ptr(array: np.ndarray):
    return array.ctypes.data_as(_DOUBLE_P)


def _flat(values, dimension: Optional[int] = None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if dimension is not None and array.size != dimension:
        raise ValueError(f"expected {dimension} values, got {array.size}")
    return array


class MDEngine:
    """One engine library, resolved by the symbol prefix."""

    def __init__(self, library: str, prefix: str):
        self.prefix = prefix
        self._lib = C.CDLL(library)
        self._calls: dict[str, Callable] = {}
        for name, (restype, argtypes) in _CALLS.items():
            symbol = f"{prefix}_md_{name}"
            try:
                function = getattr(self._lib, symbol)
            except AttributeError:
                if name in _REQUIRED:
                    raise MDError(f"{library} has no {symbol}") from None
                continue
            function.restype = restype
            function.argtypes = argtypes
            self._calls[name] = function
        if self._calls["abi_version"]() != ABI_VERSION:
            raise MDError(f"{library} speaks another MD session ABI")
        if not self._calls["available"]():
            raise MDError(f"{library} was built without its engine")
        self.features = int(self._calls["features"]())
        self.name = self._call_text("engine")
        self.integrator = self._call_text("integrator")

    def _call_text(self, name: str) -> str:
        function = self._calls.get(name)
        return function().decode() if function is not None else ""

    def has(self, call: str) -> bool:
        return call in self._calls

    def last_error(self) -> str:
        return self._calls["last_error"]().decode()

    def create(self, spec: str, time_step_ps: float = 0.001) -> "MDSession":
        config = _Config(C.sizeof(_Config), time_step_ps)
        err = C.create_string_buffer(_ERR_LEN)
        handle = self._calls["create"](spec.encode(), C.byref(config), err, _ERR_LEN)
        if not handle:
            raise MDError(err.value.decode())
        return MDSession(self, handle)


class MDSession:
    def __init__(self, engine: MDEngine, handle: int):
        self.engine = engine
        self._handle = C.c_void_p(handle)
        self.atoms = int(engine._calls["atom_count"](self._handle))
        self.dimension = 3 * self.atoms
        self._keep: list = []

    def close(self) -> None:
        if self._handle:
            self.engine._calls["destroy"](self._handle)
            self._handle = C.c_void_p(None)

    def __enter__(self) -> "MDSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _check(self, code: int) -> None:
        if code != 0:
            raise MDError(self.engine.last_error())

    def _need(self, call: str) -> Callable:
        function = self.engine._calls.get(call)
        if function is None:
            raise MDError(
                f"{self.engine.name} does not implement {self.engine.prefix}_md_{call}"
            )
        return function

    @property
    def masses(self) -> np.ndarray:
        out = np.empty(self.dimension)
        self._check(self.engine._calls["masses"](self._handle, _ptr(out), out.size))
        return out

    def set_box(self, box) -> None:
        array = _flat(box, 9)
        self._check(self.engine._calls["set_box"](self._handle, _ptr(array)))

    def set_positions(self, positions) -> None:
        array = _flat(positions, self.dimension)
        self._check(
            self.engine._calls["set_positions"](self._handle, _ptr(array), array.size)
        )

    def set_velocities(self, velocities) -> None:
        array = _flat(velocities, self.dimension)
        self._check(
            self.engine._calls["set_velocities"](self._handle, _ptr(array), array.size)
        )

    def set_time_step(self, time_step_ps: float) -> None:
        self._check(self._need("set_time_step")(self._handle, time_step_ps))

    def set_seed(self, seed: int) -> None:
        self._check(self._need("set_seed")(self._handle, seed))

    def initialize(self) -> None:
        self._check(self._need("initialize")(self._handle))

    def state(self) -> MDState:
        positions = np.empty(self.dimension)
        velocities = np.empty(self.dimension)
        forces = np.empty(self.dimension)
        info = _Info()
        info.bytes = C.sizeof(_Info)
        self._check(
            self.engine._calls["state"](
                self._handle,
                _ptr(positions),
                _ptr(velocities),
                _ptr(forces),
                self.dimension,
                C.byref(info),
            )
        )
        return MDState(
            positions,
            velocities,
            forces,
            info.potential_ev,
            info.kinetic_ev,
            info.time_ps,
            info.steps,
            info.force_evaluations,
            np.array(info.virial_ev).reshape(3, 3),
            np.array(info.box).reshape(3, 3),
        )

    def step_nve(self, nsteps: int) -> None:
        self._check(self.engine._calls["step_nve"](self._handle, nsteps))

    def step_projected(
        self,
        nsteps: int,
        kt_ev: Optional[float] = None,
        friction_per_ps: float = 0.0,
        plane: Optional[tuple] = None,
        walls: Sequence[tuple] = (),
    ) -> None:
        """Langevin steps on a hyperplane with planar walls.

        plane is (point, normal) of n . (x - point) = 0; each wall is
        (normal, offset) for normal . x <= offset. kt_ev None means no
        thermostat.
        """
        function = self._need("step_projected")
        keep = []
        thermostat = None
        if kt_ev is not None:
            thermostat = C.byref(_Thermostat(C.sizeof(_Thermostat), kt_ev, friction_per_ps))
        plane_arg = None
        if plane is not None:
            point = _flat(plane[0], self.dimension)
            normal = _flat(plane[1], self.dimension)
            keep += [point, normal]
            plane_arg = C.byref(
                _Plane(C.sizeof(_Plane), self.dimension, _ptr(point), _ptr(normal))
            )
        wall_array = None
        if len(walls) > 0:
            wall_array = (_Wall * len(walls))()
            for index, (normal, offset) in enumerate(walls):
                array = _flat(normal, self.dimension)
                keep.append(array)
                wall_array[index] = _Wall(_ptr(array), offset)
        self._check(
            function(
                self._handle, nsteps, thermostat, plane_arg, wall_array, len(walls)
            )
        )

    def set_hook(self, hook: Optional[Callable]) -> None:
        """hook(phase, step, positions, velocities, forces, masses) edits the
        arrays in place and returns 0, or nonzero to stop the step."""
        function = self._need("set_hook")
        if hook is None:
            self._check(function(self._handle, _HOOK(), None))
            self._keep = [k for k in self._keep if k[0] != "hook"]
            return

        def trampoline(_user, phase, step, dim, x, v, f, m):
            arrays = [np.ctypeslib.as_array(p, shape=(dim,)) for p in (x, v, f, m)]
            return int(hook(phase, step, *arrays))

        callback = _HOOK(trampoline)
        self._keep.append(("hook", callback))
        self._check(function(self._handle, callback, None))

    def set_boundary(self, boundary: Optional[Callable]) -> None:
        """boundary(positions, outward_normal) returns nonzero when the
        positions are outside; it writes the outward normal in place."""
        function = self._need("set_boundary")
        if boundary is None:
            self._check(function(self._handle, _BOUNDARY(), None))
            self._keep = [k for k in self._keep if k[0] != "boundary"]
            return

        def trampoline(_user, dim, x, normal):
            positions = np.ctypeslib.as_array(x, shape=(dim,))
            gradient = np.ctypeslib.as_array(normal, shape=(dim,))
            return int(boundary(positions, gradient))

        callback = _BOUNDARY(trampoline)
        self._keep.append(("boundary", callback))
        self._check(function(self._handle, callback, None))
