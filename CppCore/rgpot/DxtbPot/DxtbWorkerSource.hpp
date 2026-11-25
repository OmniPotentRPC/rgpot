#pragma once
// MIT License
// Copyright 2023--present rgpot developers

namespace rgpot::detail {

/// Defines runtime_problem(): None when this Python can run the worker,
/// otherwise the reason it cannot. The worker and the availability probe both
/// run it, so they agree on what a usable runtime is.
inline constexpr char kDxtbRuntimeCheck[] = R"DXTBPY(
DXTB_VERSION = "0.4.0"
TAD_LIBCINT_MINIMUM = (0, 1, 2)


def runtime_problem():
    import importlib
    import importlib.metadata
    import sys

    where = "Python " + sys.executable
    for module, package in (("torch", "torch"), ("tad_libcint", "tad-libcint"), ("dxtb", "dxtb")):
        try:
            importlib.import_module(module)
        except Exception as exc:
            return where + " cannot import " + package + " (" + type(exc).__name__ + ": " + str(exc) + ")"
    found = getattr(sys.modules["dxtb"], "__version__", "unknown")
    if found != DXTB_VERSION:
        return where + " has dxtb " + found + "; the dxtb engine requires dxtb " + DXTB_VERSION
    try:
        cint = importlib.metadata.version("tad-libcint")
    except importlib.metadata.PackageNotFoundError:
        cint = getattr(sys.modules["tad_libcint"], "__version__", "")
    parts = []
    for part in cint.split(".")[:3]:
        digits = len(part) - len(part.lstrip("0123456789"))
        parts.append(int(part[:digits] or 0))
    if not cint or tuple(parts) < TAD_LIBCINT_MINIMUM:
        return (
            where + " has tad-libcint " + (cint or "of unknown version")
            + "; the dxtb engine requires tad-libcint "
            + ".".join(str(n) for n in TAD_LIBCINT_MINIMUM) + " or newer"
        )
    return None
)DXTBPY";

inline constexpr char kDxtbWorkerSource[] = R"DXTBPY(
import os
import struct
import sys

_out = sys.stdout.buffer
sys.stdout = sys.stderr
_shutting_down = False

_runtime_problem = runtime_problem()
if _runtime_problem is None:
    import torch
    import dxtb

MAGIC_REQUEST = b"DXTB"
MAGIC_RESPONSE = b"DXTR"
VERSION = 1
OP_CONFIG = 1
OP_FORCE = 2
OP_HVP = 3
OP_THREADS = 4
OP_SHUTDOWN = 5
OK = 0
INVALID = 1
EVAL = 2
UNSUPPORTED = 3
FATAL = -1
ANGSTROM_TO_BOHR = 1.8897261254578281
HARTREE_TO_EV = 27.211386245988
HEADER = struct.Struct("<4sIIII")


class ProtocolError(Exception):
    pass


class InvalidError(Exception):
    pass


def read_exact(size):
    data = sys.stdin.buffer.read(size)
    if len(data) != size:
        raise EOFError
    return data


def response(status, payload=b""):
    try:
        _out.write(HEADER.pack(MAGIC_RESPONSE, VERSION, status & 0xffffffff, 0, len(payload)))
        _out.write(payload)
        _out.flush()
    except (BrokenPipeError, ConnectionResetError):
        # The peer has gone. A shutdown may exit cleanly; a live call must not.
        os._exit(0 if _shutting_down else 1)


def error(status, exc):
    response(status, (type(exc).__name__ + ": " + str(exc)).encode("utf-8"))


config = None
calculator = None
composition = None


def set_torch_threads(intra, inter):
    if inter > 0 and torch.get_num_interop_threads() != inter:
        torch.set_num_interop_threads(inter)
    if intra > 0 and torch.get_num_threads() != intra:
        torch.set_num_threads(intra)


def apply_config(payload):
    global config, calculator, composition
    layout = struct.Struct("<Iii3dii")
    if len(payload) != layout.size:
        raise ProtocolError("invalid configuration payload")
    method, uhf, maxiter, charge, accuracy, temperature, intra, inter = layout.unpack(payload)
    if method not in (1, 2):
        raise InvalidError("method must be GFN1 or GFN2")
    if uhf < 0 or maxiter < 1 or accuracy <= 0.0 or temperature < 0.0:
        raise InvalidError("invalid dxtb configuration")
    set_torch_threads(intra, inter)
    config = (method, uhf, maxiter, charge, accuracy, temperature)
    calculator = None
    composition = None


def get_calculator(numbers):
    global calculator, composition
    key = tuple(int(z) for z in numbers.tolist())
    if calculator is None or key != composition:
        method, _uhf, maxiter, _charge, accuracy, temperature = config
        opts = {
            "scf_mode": "full",
            "int_driver": "libcint",
            "f_atol": 1.0e-11 * accuracy,
            "x_atol": 1.0e-11 * accuracy,
            "maxiter": maxiter,
            "mixer": "simple",
            "damp": 0.3,
            "force_convergence": True,
            "fermi_etemp": temperature,
            "cache_enabled": False,
            "verbosity": 0,
        }
        par = dxtb.GFN1_XTB if method == 1 else dxtb.GFN2_XTB
        calculator = dxtb.Calculator(
            numbers, par, opts=opts, dtype=torch.float64, device=torch.device("cpu")
        )
        composition = key
    calculator.reset()
    return calculator


def decode_geometry(natoms, payload, with_direction):
    base = 4 * natoms + 8 * 3 * natoms
    expected = base + (8 * 3 * natoms if with_direction else 0)
    if natoms < 1 or len(payload) != expected:
        raise ProtocolError("invalid geometry payload")
    numbers = torch.tensor(
        struct.unpack_from("<" + "i" * natoms, payload, 0), dtype=torch.int64
    )
    if bool(torch.any(numbers < 1)) or bool(torch.any(numbers > 86)):
        raise InvalidError("atomic number is outside dxtb's supported range")
    offset = 4 * natoms
    positions = torch.tensor(
        struct.unpack_from("<" + "d" * (3 * natoms), payload, offset),
        dtype=torch.float64,
    ).reshape(natoms, 3)
    positions = (positions * ANGSTROM_TO_BOHR).requires_grad_(True)
    direction = None
    if with_direction:
        offset += 8 * 3 * natoms
        direction = torch.tensor(
            struct.unpack_from("<" + "d" * (3 * natoms), payload, offset),
            dtype=torch.float64,
        ).reshape(natoms, 3)
        direction = direction * ANGSTROM_TO_BOHR
    return numbers, positions, direction


def evaluate(natoms, payload, want_hvp):
    numbers, positions, direction = decode_geometry(natoms, payload, want_hvp)
    method, uhf, _maxiter, charge, _accuracy, _temperature = config
    del method
    calc = get_calculator(numbers)
    energy = calc.energy(positions, chrg=charge, spin=uhf)
    gradient = torch.autograd.grad(
        energy.sum(), positions, create_graph=want_hvp
    )[0]
    if want_hvp:
        product = torch.autograd.grad(
            gradient, positions, grad_outputs=direction
        )[0]
        values = (
            product.detach().reshape(-1)
            * HARTREE_TO_EV
            * ANGSTROM_TO_BOHR
        )
        return struct.pack("<" + "d" * values.numel(), *values.tolist())
    forces = -gradient.detach().reshape(-1) * HARTREE_TO_EV * ANGSTROM_TO_BOHR
    return struct.pack(
        "<" + "d" * (1 + forces.numel()),
        float(energy.detach().reshape(-1)[0]) * HARTREE_TO_EV,
        *forces.tolist(),
    )


while True:
    try:
        raw = read_exact(HEADER.size)
    except EOFError:
        break
    except (BrokenPipeError, ConnectionResetError):
        os._exit(0 if _shutting_down else 1)
    try:
        magic, version, op, natoms, size = HEADER.unpack(raw)
        if magic != MAGIC_REQUEST or version != VERSION:
            raise ProtocolError("invalid request header")
        payload = read_exact(size)
        if op == OP_SHUTDOWN:
            if size != 0:
                raise ProtocolError("shutdown has a payload")
            _shutting_down = True
            response(OK)
            # Skip interpreter teardown so it cannot write to the socket after
            # the host has finished reading this acknowledgement.
            os._exit(0)
        if _runtime_problem is not None:
            response(FATAL, _runtime_problem.encode("utf-8"))
            continue
        if op == OP_CONFIG:
            apply_config(payload)
            response(OK)
        elif op == OP_FORCE:
            if config is None:
                raise ProtocolError("worker is not configured")
            response(OK, evaluate(natoms, payload, False))
        elif op == OP_HVP:
            if config is None:
                raise ProtocolError("worker is not configured")
            response(OK, evaluate(natoms, payload, True))
        elif op == OP_THREADS:
            if len(payload) != 8:
                raise ProtocolError("invalid thread payload")
            intra, inter = struct.unpack("<ii", payload)
            set_torch_threads(intra, inter)
            response(OK)
        else:
            raise ProtocolError("unknown operation")
    except (BrokenPipeError, ConnectionResetError):
        os._exit(0 if _shutting_down else 1)
    except ProtocolError as exc:
        error(FATAL, exc)
    except InvalidError as exc:
        error(INVALID, exc)
    except Exception as exc:
        error(EVAL, exc)
os._exit(0)
)DXTBPY";

} // namespace rgpot::detail
