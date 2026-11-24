"""ASE calculator backed by an rgpot Cap'n Proto potential server.

The connection belongs to this calculator (F3). A second thread that
enters :meth:`calculate` or :meth:`close` while the first is inside
raises ``RuntimeError``.

Units follow the Cap'n Proto ``ForceInput`` defaults (Angstrom / eV),
which are ASE's units.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    PropertyNotImplementedError,
    all_changes,
)

from rgpot.ase._registry import register_calculator as _register_calculator
from rgpot.ase.calculator import _voigt
from rgpot.ase.claim import ConnectionClaim

PROTOCOL_FAMILY = "rgpot.potentials"
PROTOCOL_MAJOR = 1
PROTOCOL_MINOR = 2
# The minor that introduced the orbital-key fields. It is a fixed number so a
# later PROTOCOL_MINOR bump does not raise the floor for keyed requests.
ORBITAL_KEY_INTRODUCED_MINOR = 2


class ProtocolMinorMismatchError(RuntimeError):
    """The server advertises a protocol minor below what a request needs.

    ``required_minor`` is the minor the request needs, ``server_minor`` the
    one the server advertised and ``client_minor`` the one this client
    implements. ``feature`` names the request field that set the floor.
    """

    def __init__(
        self,
        feature: str,
        required_minor: int,
        server_minor: int,
        client_minor: int,
    ) -> None:
        self.feature = feature
        self.required_minor = int(required_minor)
        self.server_minor = int(server_minor)
        self.client_minor = int(client_minor)
        super().__init__(
            f"ForceInput {feature} requires RPC protocol minor "
            f"{self.required_minor}, but the server advertises "
            f"{self.server_minor} (the client implements "
            f"{self.client_minor})"
        )


def _capnp() -> Any:
    """Import capnp on first use so the module imports without the package."""
    import capnp  # pyright: ignore[reportMissingImports]

    return capnp


def _schema_path() -> Path:
    here = Path(__file__).resolve().parent / "Potentials.capnp"
    if here.is_file():
        return here
    repo = (
        Path(__file__).resolve().parents[3]
        / "CppCore"
        / "rgpot"
        / "rpc"
        / "Potentials.capnp"
    )
    if repo.is_file():
        return repo
    raise FileNotFoundError("Potentials.capnp is not installed with rgpot.ase")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return int(sock.getsockname()[1])


class RgpotRpcCalculator(Calculator):
    """ASE calculator that evaluates a potential through potserv.

    Parameters
    ----------
    host, port:
        Address of a running potserv instance.
    server_proc:
        Optional managed process this calculator stops on :meth:`close`.
        Prefer :meth:`spawn`.
    orbital_key:
        Optional image or bead key. ``atoms.info["rgpot_orbital_key"]``
        overrides it for one calculation.
    """

    implemented_properties = [  # noqa: RUF012
        "energy",
        "free_energy",
        "forces",
    ]
    # The connection is a session (F3), not the pure protocol.
    reentrant = False

    def __init__(
        self,
        host: str = "localhost",
        port: int = 12345,
        *,
        server_proc: subprocess.Popen[bytes] | None = None,
        orbital_key: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._host = host
        self._port = int(port)
        self._server_proc = server_proc
        self._orbital_key = None if orbital_key is None else int(orbital_key)
        self._server_protocol_minor: int | None = None
        self._claim = ConnectionClaim("RgpotRpcCalculator")

        self._schema = _capnp().load(str(_schema_path()))
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._kj = _capnp().kj_loop()
        self._closed = False
        self._pot: Any = None
        self._server_has_stress = False
        self._client: Any = None
        self._connection: Any = None
        self._loop.run_until_complete(self._kj.__aenter__())
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException:
            self.close()
            raise

    @classmethod
    def spawn(
        cls,
        server_bin: str | None,
        potential: str = "CuH2",
        port: int | None = None,
        *,
        startup_timeout: float = 30.0,
        **kwargs: Any,
    ) -> RgpotRpcCalculator:
        """Launch a managed potserv and return a calculator bound to it."""
        del startup_timeout
        if server_bin is None:
            server_bin = os.environ.get("RGPOT_POTSERV_BIN")
        if not server_bin or not Path(server_bin).exists():
            raise FileNotFoundError(
                f"potserv binary not found: {server_bin!r} "
                "(set the path or RGPOT_POTSERV_BIN)"
            )
        chosen = port or _free_port()
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            [str(server_bin), str(chosen), potential],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            return cls(host="localhost", port=chosen, server_proc=proc, **kwargs)
        except Exception:
            proc.kill()
            proc.wait()
            raise

    def close(self) -> None:
        """Disconnect and stop a managed server, if any."""
        with self._claim:
            if not self._closed:
                self._closed = True
                self._pot = None
                self._client = None
                self._connection = None
                import gc
                gc.collect()
                try:
                    self._loop.run_until_complete(self._kj.__aexit__(None, None, None))
                except Exception:  # noqa: BLE001, S110
                    pass
                self._loop.close()
            if self._server_proc is not None:
                self._server_proc.kill()
                self._server_proc.wait()
                self._server_proc = None

    def __enter__(self) -> "RgpotRpcCalculator":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def _connect(self) -> None:
        last_err: OSError | None = None
        for _ in range(120):
            if self._server_proc is not None and self._server_proc.poll() is not None:
                out, err = self._server_proc.communicate()
                msg = (err or out or b"").decode(errors="replace").strip()
                raise RuntimeError(
                    f"RgpotRpcCalculator: potserv exited with code "
                    f"{self._server_proc.returncode}: {msg}"
                )
            try:
                connection = await _capnp().AsyncIoStream.create_connection(
                    host=self._host, port=self._port
                )
                client = _capnp().TwoPartyClient(connection)
                self._pot = client.bootstrap().cast_as(self._schema.Potential)
                await self._refresh_capabilities_async()
                return
            except OSError as exc:
                last_err = exc
                await asyncio.sleep(0.25)
        raise RuntimeError(
            "RgpotRpcCalculator: could not connect to "
            f"{self._host}:{self._port}: {last_err}"
        )

    async def _refresh_capabilities_async(self) -> None:
        try:
            reply = await self._pot.getCapabilities()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RgpotRpcCalculator: getCapabilities RPC failed"
            ) from exc
        operations = {
            str(getattr(operation, "name", operation)).rsplit(".", 1)[-1].lower()
            for operation in reply.capabilities.operations
        }
        self._server_has_stress = "stress" in operations
        properties = ["energy", "free_energy", "forces"]
        if self._server_has_stress:
            properties.append("stress")
        self.implemented_properties = properties

    async def _calculate_async(
        self,
        positions: Any,
        numbers: Any,
        box: Any,
        orbital_key: int | None = None,
        *,
        request_stress: bool = False,
        accept_stress: bool = False,
    ) -> dict[str, Any]:
        if orbital_key is not None:
            await self._check_server_protocol(
                ORBITAL_KEY_INTRODUCED_MINOR, "orbital key"
            )
        fip = self._schema.ForceInput.new_message()
        fip.pos = np.asarray(positions, dtype=float).ravel().tolist()
        fip.atmnrs = [int(z) for z in numbers]
        fip.box = np.asarray(box, dtype=float).ravel()[:9].tolist()
        if orbital_key is not None:
            fip.hasOrbitalKey = True
            fip.orbitalKey = int(orbital_key)
        try:
            reply = (
                await self._pot.calculateStress(fip)
                if request_stress
                else await self._pot.calculate(fip)
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("RgpotRpcCalculator: calculate RPC failed") from exc
        result = reply.result
        energy = float(result.energy)
        evaluated: dict[str, Any] = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.array(result.forces, dtype=float).reshape(-1, 3),
        }
        if accept_stress:
            tensor = np.asarray(result.stress, dtype=float)
            if tensor.size != 9:
                if request_stress:
                    raise PropertyNotImplementedError(
                        "RgpotRpcCalculator: the connected server potential "
                        "returned no stress"
                    )
            else:
                evaluated["stress"] = _voigt(tensor)
        return evaluated

    def get_property(
        self,
        name: str,
        atoms: Any = None,
        allow_calculation: bool = True,
    ) -> Any:
        if name == "stress":
            with self._claim:
                try:
                    self._loop.run_until_complete(self._refresh_capabilities_async())
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError("RgpotRpcCalculator: RPC preflight failed") from exc
                if not self._server_has_stress:
                    raise PropertyNotImplementedError(
                        "RgpotRpcCalculator: the connected server potential "
                        "does not provide stress"
                    )
        return super().get_property(name, atoms, allow_calculation)

    async def _check_server_protocol(
        self, required_minor: int, feature: str
    ) -> None:
        minor = getattr(self, "_server_protocol_minor", None)
        if minor is None:
            response = await self._pot.getCapabilities()
            capabilities = response.capabilities
            family = str(capabilities.protocolFamily)
            major = int(capabilities.protocolMajor)
            if family != PROTOCOL_FAMILY or major != PROTOCOL_MAJOR:
                raise RuntimeError(
                    f"RPC protocol {family!r} major {major} is not "
                    f"{PROTOCOL_FAMILY!r} major {PROTOCOL_MAJOR}"
                )
            minor = int(capabilities.protocolMinor)
            self._server_protocol_minor = minor
        if minor < required_minor:
            raise ProtocolMinorMismatchError(
                feature, required_minor, minor, PROTOCOL_MINOR
            )

    def select_orbitals(self, key: int) -> None:
        """Select the image or bead whose orbitals following calls use."""
        self._orbital_key = int(key)

    async def _neighbours_async(
        self,
        positions: Any,
        numbers: Any,
        box: Any,
        pbc: Any,
        *,
        cutoff: float,
        skin: float,
        full: bool,
        strict: bool,
        shifts: bool,
        vectors: bool,
        distances: bool,
    ) -> dict[str, np.ndarray]:
        fip = self._schema.ForceInput.new_message()
        fip.pos = np.asarray(positions, dtype=float).ravel().tolist()
        fip.atmnrs = [int(z) for z in numbers]
        fip.box = np.asarray(box, dtype=float).ravel()[:9].tolist()
        req = self._schema.NeighborRequest.new_message()
        req.cutoff = float(cutoff)
        req.skin = float(skin)
        req.format = 1 if full else 0
        req.cut = 0 if strict else 1
        req.periodic = [bool(flag) for flag in pbc]
        req.shifts = bool(shifts)
        req.vectors = bool(vectors)
        req.distances = bool(distances)
        req.mode = 1
        reply = (await self._pot.neighbours(fip, req)).neighbors
        if int(reply.status) != 0:
            raise ValueError(str(reply.message))
        out: dict[str, np.ndarray] = {
            "first": np.asarray(reply.first, dtype=np.int32),
            "second": np.asarray(reply.second, dtype=np.int32),
        }
        if shifts:
            out["shift"] = np.asarray(reply.shift, dtype=np.int32).reshape(-1, 3)
        if vectors:
            out["vector"] = np.asarray(reply.vector, dtype=np.float64).reshape(-1, 3)
        if distances:
            out["distance"] = np.asarray(reply.distance, dtype=np.float64)
        return out

    def neighbours(
        self,
        atoms: Any = None,
        *,
        cutoff: float = 0.0,
        skin: float = 0.0,
        full: bool = False,
        strict: bool = True,
        shifts: bool = True,
        vectors: bool = True,
        distances: bool = True,
    ) -> dict[str, np.ndarray]:
        """Request the remote potential's neighbour list for ``atoms``."""
        with self._claim:
            frame = atoms if atoms is not None else self.atoms
            if frame is None:
                raise ValueError("neighbours requires an Atoms object")
            pbc_arr = np.asarray(frame.pbc, dtype=bool).reshape(-1)
            if pbc_arr.size == 1:
                pbc_flags = [bool(pbc_arr[0])] * 3
            elif pbc_arr.size == 3:
                pbc_flags = [bool(x) for x in pbc_arr]
            else:
                raise ValueError(f"pbc must have 1 or 3 entries, got {int(pbc_arr.size)}")
            cell = frame.cell
            if cell.rank == 3:
                box = np.asarray(cell.array, dtype=float)
            else:
                box = np.eye(3, dtype=float) * 100.0
            return self._loop.run_until_complete(
                self._neighbours_async(
                    frame.get_positions(),
                    frame.get_atomic_numbers(),
                    box,
                    pbc_flags,
                    cutoff=cutoff,
                    skin=skin,
                    full=full,
                    strict=strict,
                    shifts=shifts,
                    vectors=vectors,
                    distances=distances,
                )
            )

    def calculate(
        self,
        atoms: Any = None,
        properties: Any = ("energy",),
        system_changes: Any = all_changes,
    ) -> None:
        with self._claim:
            try:
                self._loop.run_until_complete(self._refresh_capabilities_async())
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("RgpotRpcCalculator: RPC preflight failed") from exc
            super().calculate(atoms, properties, system_changes)
            frame = self.atoms
            if frame is None:
                raise RuntimeError("RgpotRpcCalculator: ASE provided no atoms frame")
            cell = frame.cell
            wants_stress = "stress" in set(properties)
            fully_periodic = cell.rank == 3 and bool(np.all(frame.pbc))
            if wants_stress and not self._server_has_stress:
                raise PropertyNotImplementedError(
                    "RgpotRpcCalculator: the connected server potential "
                    "does not provide stress"
                )
            if wants_stress and not fully_periodic:
                raise PropertyNotImplementedError(
                    "RgpotRpcCalculator: stress requires a fully periodic "
                    "three-dimensional cell"
                )
            if cell.rank == 3:
                box = np.asarray(cell.array)
            else:
                box = np.eye(3) * 100.0
            orbital_key = frame.info.get("rgpot_orbital_key", self._orbital_key)
            try:
                evaluated = self._loop.run_until_complete(
                    self._calculate_async(
                        frame.get_positions(),
                        frame.get_atomic_numbers(),
                        box,
                        orbital_key,
                        request_stress=wants_stress,
                        accept_stress=self._server_has_stress and fully_periodic,
                    )
                )
            except (
                PropertyNotImplementedError,
                CalculatorError,
                ProtocolMinorMismatchError,
            ):
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("RgpotRpcCalculator: RPC calculate failed") from exc
            self.results.update(evaluated)

_register_calculator("rgpot_rpc", RgpotRpcCalculator)
