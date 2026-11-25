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

import pytest

pytestmark = pytest.mark.rpc

_PRELUDE = """
import asyncio
import gc
import os
import sys

import numpy as np
from ase import Atoms

from rgpot.ase import RgpotRpcCalculator

host, port = sys.argv[1], int(sys.argv[2])
atoms = Atoms(
    "Ar3",
    positions=[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 3.3, 0.0]],
    cell=[30.0, 30.0, 30.0],
    pbc=False,
)


class ExitSpy:
    # Records which handles are still alive when the kj loop is left.
    def __init__(self, calc):
        self.calc = calc
        self.inner = calc._kj
        self.seen = []
        calc._kj = self

    async def __aenter__(self):
        return await self.inner.__aenter__()

    async def __aexit__(self, *exc):
        c = self.calc
        self.seen.append(
            (c._pot, getattr(c, "_client", None), getattr(c, "_connection", None))
        )
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
        assert spy.seen == [(None, None, None)], spy.seen
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
        assert spy.seen == [(None, None, None)], spy.seen
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
        free = __import__("socket").socket()
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
}


@pytest.mark.parametrize("scenario", sorted(_SCENARIOS))
def test_rpc_close_ordering(scenario: str, potserv_factory) -> None:
    server = potserv_factory("LJ")
    script = _PRELUDE + textwrap.dedent(_SCENARIOS[scenario]) + "\nprint('done')\n"
    proc = subprocess.run(
        [
            sys.executable,
            "-X",
            "faulthandler",
            "-W",
            "error::RuntimeWarning",
            "-c",
            script,
            server.host,
            str(server.port),
        ],
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


def test_rpc_spawned_server_stops_on_close(potserv_server) -> None:
    from rgpot.ase import RgpotRpcCalculator

    calc = RgpotRpcCalculator.spawn(str(potserv_server.binary), potential="LJ")
    proc = calc._server_proc
    assert proc is not None
    calc.close()
    calc.close()
    assert proc.poll() is not None
    assert calc._server_proc is None
