"""ASE calculator backed by an rgpot Cap'n Proto potential server.

The connection belongs to this calculator (F3). A second thread that
enters :meth:`calculate` or :meth:`close` while the first is inside
raises ``RuntimeError``.

Units follow the Cap'n Proto ``ForceInput`` defaults (Angstrom / eV),
which are ASE's units.
"""

from __future__ import annotations

import atexit
import asyncio
import os
import socket
import subprocess
import weakref
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    PropertyNotImplementedError,
    all_changes,
)

from rgpot._schema import load_potentials_schema, schema_path
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
    return schema_path()


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

        self._schema = load_potentials_schema()
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._kj = _capnp().kj_loop()
        self._closed = False
        self._pot: Any = None
        self._client: Any = None
        self._connection: Any = None
        self._exit_hook: Any = None
        self._server_has_stress = False
        self._loop.run_until_complete(self._kj.__aenter__())
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException:
            self.close()
            raise
        # Interpreter exit must run close() while this object is still
        # intact. The hook holds a weak reference so a dropped calculator
        # can still be collected and shut down from __del__.
        self._register_exit_hook()

    @staticmethod
    def _stop_server_proc(proc: subprocess.Popen[bytes]) -> None:
        try:
            proc.kill()
            proc.wait()
        finally:
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()

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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return cls(host="localhost", port=chosen, server_proc=proc, **kwargs)
        except BaseException:
            cls._stop_server_proc(proc)
            raise

    def _register_exit_hook(self) -> None:
        ref = weakref.ref(self)

        def hook() -> None:
            calc = ref()
            if calc is None:
                return
            try:
                calc.close()
            except BaseException:
                return

        self._exit_hook = hook
        atexit.register(hook)

    def _unregister_exit_hook(self) -> None:
        hook = getattr(self, "_exit_hook", None)
        if hook is None:
            return
        atexit.unregister(hook)
        self._exit_hook = None

    def _drop_rpc_clients(self) -> None:
        """Drop capability clients while the KJ loop object still exists."""
        self._pot = None
        if hasattr(self, "_client"):
            self._client = None
        if hasattr(self, "_connection"):
            self._connection = None

    def close(self) -> None:
        """Disconnect and stop a managed server, if any."""
        if not hasattr(self, "_claim") or not hasattr(self, "_closed"):
            self._unregister_exit_hook()
            return
        with self._claim:
            try:
                if not self._closed:
                    self._closed = True
                    # Drop Python-side capability references before kj teardown.
                    # A client destroyed after the KJ loop aborts the process.
                    self._drop_rpc_clients()
                    if not self._loop.is_closed():
                        exit_coro: Any = None
                        try:
                            exit_coro = self._kj.__aexit__(None, None, None)
                            self._loop.run_until_complete(exit_coro)
                        except Exception:  # noqa: BLE001
                            # If teardown fails after creating the coroutine
                            # (for example because the loop closes), close the
                            # coroutine to avoid "never awaited" warnings.
                            if exit_coro is not None:
                                exit_coro.close()
                        except BaseException:
                            if exit_coro is not None:
                                exit_coro.close()
                            raise
                        finally:
                            if not self._loop.is_closed():
                                self._loop.close()
            finally:
                proc = self._server_proc
                if proc is not None:
                    self._server_proc = None
                    self._stop_server_proc(proc)
        self._unregister_exit_hook()

    def __del__(self) -> None:
        # Collection can run after the asyncio loop's finalizer has closed
        # that loop and before the KJ loop object is cleared. Drop clients
        # first, then use the same shutdown as an explicit close().
        try:
            self._drop_rpc_clients()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.close()
        except BaseException:
            return

    def __enter__(self) -> "RgpotRpcCalculator":
        return self

    def __exit__(self, *exc: object) -> None:
        if exc and exc[0] is not None:
            try:
                self.close()
            except BaseException:  # noqa: BLE001, S110
                pass
        else:
            self.close()

    async def _connect(self) -> None:
        last_err: OSError | None = None
        for _ in range(120):
            try:
                connection = await _capnp().AsyncIoStream.create_connection(
                    host=self._host, port=self._port
                )
                client = _capnp().TwoPartyClient(connection)
                self._connection = connection
                self._client = client
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
