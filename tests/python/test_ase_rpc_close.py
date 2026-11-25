# Copyright 2023--present rgpot developers

"""RgpotRpcCalculator.close() releases capnp handles before the kj loop exits.

Every scenario runs in a child interpreter against the live potserv from the
fixture. A native abort in the child then fails the test with its output
instead of killing the pytest process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.rpc

_PRELUDE = """
import asyncio
import gc
import os
import socket
import subprocess
import sys
import threading

import numpy as np
from ase import Atoms

from rgpot.ase import RgpotRpcCalculator

host, port, potserv_bin = sys.argv[1], int(sys.argv[2]), sys.argv[3]
atoms = Atoms(
    "Ar3",
    positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]],
    cell=[30.0, 30.0, 30.0],
    pbc=False,
)


class ExitSpy:
    # Records capability handle and loop state when the kj loop is left.
    def __init__(self, calc):
        self.calc = calc
        self.inner = calc._kj
        self.seen = []
        calc._kj = self

    async def __aenter__(self):
        return await self.inner.__aenter__()

    async def __aexit__(self, *exc):
        c = self.calc
        self.seen.append((c._pot, c._closed, c._loop.is_closed()))
        return await self.inner.__aexit__(*exc)


def fresh():
    calc = RgpotRpcCalculator(host=host, port=port)
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc
"""

_SCENARIOS = {
    "live_capability": """
        calc = fresh()
        spy = ExitSpy(calc)
        held = calc._pot
        assert held is not None
        calc.close()
        assert spy.seen == [(None, True, False)], spy.seen
        assert calc._loop.is_closed()
        del held
        gc.collect()
        gc.collect()
    """,
    "request_in_flight": """
        calc = fresh()
        spy = ExitSpy(calc)
        box = np.eye(3) * 30.0
        tasks = [
            calc._loop.create_task(
                calc._calculate_async(
                    atoms.get_positions(), atoms.get_atomic_numbers(), box
                )
            )
            for _ in range(8)
        ]
        calc._loop.run_until_complete(asyncio.sleep(0))
        calc.close()
        assert spy.seen == [(None, True, False)], spy.seen
        assert calc._loop.is_closed()
        assert all(t.done() for t in tasks)
        gc.collect()
    """,
    "double_close": """
        calc = fresh()
        spy = ExitSpy(calc)
        calc.close()
        calc.close()
        with calc:
            pass
        assert len(spy.seen) == 1, spy.seen
        assert spy.seen == [(None, True, False)], spy.seen
        assert calc._loop.is_closed()
    """,
    "close_after_loop_closed": """
        calc = fresh()
        spy = ExitSpy(calc)
        calc._loop.close()
        calc.close()
        calc.close()
        assert spy.seen == [], spy.seen
        assert calc._loop.is_closed()
    """,
    "masked_loop_closed_during_teardown": """
        calc = fresh()
        spy = ExitSpy(calc)
        original_run = calc._loop.run_until_complete

        def close_then_raise(coro):
            calc._loop.close()
            raise RuntimeError("Event loop is closed")

        calc._loop.run_until_complete = close_then_raise
        calc.close()
        calc.close()
        calc._loop.run_until_complete = original_run
        assert spy.seen == [], spy.seen
        assert calc._loop.is_closed()
    """,
    "close_then_gc": """
        calc = fresh()
        calc.close()
        atoms.calc = None
        del calc
        gc.collect()
        gc.collect()
        calc = fresh()
        atoms.calc = None
        calc.close()
        del calc
        gc.collect()
    """,
    "close_after_failed_connect": """
        free = socket.socket()
        free.bind(("localhost", 0))
        dead = free.getsockname()[1]
        free.close()
        import rgpot.ase.rpc as rpc

        orig = rpc.asyncio.sleep

        async def quick(_delay, *a, **k):
            return await orig(0)

        rpc.asyncio.sleep = quick
        try:
            RgpotRpcCalculator(host="localhost", port=dead)
        except RuntimeError as exc:
            assert "could not connect" in str(exc), exc
        else:
            raise AssertionError("connect to a dead port must fail")
        finally:
            rpc.asyncio.sleep = orig
        gc.collect()
    """,
    "exit_does_not_mask_body_exception_when_claimed": """
        calc = fresh()
        entered = threading.Event()
        release = threading.Event()

        def hold_claim():
            with calc._claim:
                entered.set()
                release.wait(timeout=30)

        holder = threading.Thread(target=hold_claim)
        holder.start()
        try:
            assert entered.wait(timeout=30)
            try:
                with calc:
                    raise ValueError("body failure")
            except ValueError as exc:
                assert str(exc) == "body failure", exc

            try:
                with calc:
                    pass
            except RuntimeError as exc:
                assert "in use" in str(exc), exc
            else:
                raise AssertionError("normal __exit__ must surface claim RuntimeError")
        finally:
            release.set()
            holder.join(timeout=30)
        calc.close()
        assert calc._loop.is_closed()
    """,
    "keyboard_interrupt_from_kj_aexit_closes_loop_and_server": """
        calc = RgpotRpcCalculator.spawn(potserv_bin, potential="LJ")
        proc = calc._server_proc
        assert proc is not None
        inner = calc._kj

        class InterruptingKj:
            async def __aexit__(self, *exc):
                await inner.__aexit__(*exc)
                raise KeyboardInterrupt("ctrl-c")

        calc._kj = InterruptingKj()
        try:
            calc.close()
        except KeyboardInterrupt as exc:
            assert str(exc) == "ctrl-c", exc
        else:
            raise AssertionError("KeyboardInterrupt must propagate")
        assert calc._loop.is_closed()
        assert calc._server_proc is None
        assert proc.poll() is not None
        assert not calc._claim._busy
    """,
    "cancelled_error_from_kj_aexit_closes_loop_and_server": """
        calc = RgpotRpcCalculator.spawn(potserv_bin, potential="LJ")
        proc = calc._server_proc
        assert proc is not None
        inner = calc._kj

        class CancellingKj:
            async def __aexit__(self, *exc):
                await inner.__aexit__(*exc)
                raise asyncio.CancelledError("cancelled")

        calc._kj = CancellingKj()
        try:
            calc.close()
        except asyncio.CancelledError as exc:
            assert str(exc) == "cancelled", exc
        else:
            raise AssertionError("CancelledError must propagate")
        assert calc._loop.is_closed()
        assert calc._server_proc is None
        assert proc.poll() is not None
        assert not calc._claim._busy
    """,
    "cancelled_error_before_await_closes_coro_loop_and_server": """
        calc = RgpotRpcCalculator.spawn(potserv_bin, potential="LJ")
        proc = calc._server_proc
        assert proc is not None
        real_kj = calc._kj
        real_run = calc._loop.run_until_complete

        class DummyKj:
            async def __aexit__(self, *exc):
                return False

        def raise_before_await(_coro):
            raise asyncio.CancelledError("aborted")

        # Exit the real kj context before substituting a dummy coroutine so
        # the real C++ context tears down cleanly while we exercise the
        # unawaited-coroutine cleanup path on BaseException.
        calc._pot = None
        real_run(real_kj.__aexit__(None, None, None))
        calc._kj = DummyKj()
        calc._loop.run_until_complete = raise_before_await
        try:
            try:
                calc.close()
            except asyncio.CancelledError as exc:
                assert str(exc) == "aborted", exc
            else:
                raise AssertionError("CancelledError must propagate")
        finally:
            calc._loop.run_until_complete = real_run
        assert calc._loop.is_closed()
        assert calc._server_proc is None
        assert proc.poll() is not None
        assert not calc._claim._busy
        gc.collect()
    """,
    "exit_does_not_mask_body_exception_on_kj_cancel": """
        calc = RgpotRpcCalculator.spawn(potserv_bin, potential="LJ")
        proc = calc._server_proc
        assert proc is not None
        inner = calc._kj

        class CancellingKj:
            async def __aexit__(self, *exc):
                await inner.__aexit__(*exc)
                raise asyncio.CancelledError("cancelled")

        calc._kj = CancellingKj()
        try:
            with calc:
                raise ValueError("body error")
        except ValueError as exc:
            assert str(exc) == "body error", exc
        assert calc._loop.is_closed()
        assert calc._server_proc is None
        assert proc.poll() is not None
        assert not calc._claim._busy
    """,
    "explicit_server_proc_pipes_closed": """
        sock = socket.socket()
        sock.bind(("localhost", 0))
        chosen = sock.getsockname()[1]
        sock.close()
        proc = subprocess.Popen(
            [potserv_bin, str(chosen), "LJ"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            calc = RgpotRpcCalculator(host="localhost", port=chosen, server_proc=proc)
            atoms.calc = calc
            atoms.get_potential_energy()
            calc.close()
            assert proc.poll() is not None
            assert proc.stdout.closed
            assert proc.stderr.closed
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()
        del proc
        gc.collect()
    """,
}


_ALLOW_CLOSED_LOOP_RESOURCE_WARNING = frozenset(
    {
        "close_after_loop_closed",
        "masked_loop_closed_during_teardown",
    }
)


def _run_child(
    script: str,
    host: str,
    port: int,
    binary: Path,
    *,
    check_resource_warning: bool = True,
) -> None:
    cmd = [
        sys.executable,
        "-X",
        "faulthandler",
        "-W",
        "error::RuntimeWarning",
    ]
    if check_resource_warning:
        cmd.extend(["-W", "error::ResourceWarning"])
    cmd.extend(["-c", script, host, str(port), str(binary)])
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert proc.stdout.strip().endswith("done"), output
    assert "Exception ignored" not in proc.stderr, output
    assert "terminate" not in proc.stderr, output


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_rpc_close_ordering(scenario: str, potserv_factory) -> None:
    server = potserv_factory("LJ")
    script = _PRELUDE + textwrap.dedent(_SCENARIOS[scenario]) + "\nprint('done')\n"
    _run_child(
        script,
        server.host,
        server.port,
        server.binary,
        check_resource_warning=scenario not in _ALLOW_CLOSED_LOOP_RESOURCE_WARNING,
    )


def test_rpc_spawned_server_stops_on_close(potserv_server) -> None:
    from rgpot.ase import RgpotRpcCalculator

    calc = RgpotRpcCalculator.spawn(str(potserv_server.binary), potential="LJ")
    proc = calc._server_proc
    try:
        assert proc is not None
        assert proc.stdout is None
        assert proc.stderr is None
        calc.close()
        calc.close()
        assert proc.poll() is not None
        assert calc._server_proc is None
    finally:
        calc.close()
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                if pipe is not None:
                    pipe.close()


def test_rpc_spawn_handles_noisy_stdio_without_blocking_or_warning(
    tmp_path: Path, potserv_bin: Path
) -> None:
    wrapper = tmp_path / "noisy_potserv.py"
    wrapper.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import os
            import sys

            chunk = b"noisy potserv output\\n" * 16384
            os.write(1, chunk)
            os.write(2, chunk)
            os.execv({str(potserv_bin)!r}, [{str(potserv_bin)!r}, *sys.argv[1:]])
            """
        ),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    script = (
        _PRELUDE
        + textwrap.dedent(
            f"""
            calc = RgpotRpcCalculator.spawn({str(wrapper)!r}, potential="LJ")
            proc = calc._server_proc
            try:
                assert proc is not None
                assert proc.stdout is None
                assert proc.stderr is None
                atoms.calc = calc
                atoms.get_potential_energy()
                calc.close()
                assert proc.poll() is not None
            finally:
                calc.close()
                if proc is not None and proc.poll() is None:
                    proc.kill()
                    proc.wait()
            del proc
            gc.collect()
            print("done")
            """
        )
    )
    _run_child(script, "localhost", 0, potserv_bin)
