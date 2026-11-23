# Copyright 2023--present rgpot developers

"""Regression test for RPC calculator close-loop teardown."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Mapping

import pytest


def _run_python(script: str, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
        env=dict(env) if env is not None else os.environ.copy(),
    )


@pytest.mark.rpc
def test_rpc_spawn_close_is_idempotent_and_does_not_abort() -> None:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    proc = _run_python(
        """
        import os

        from ase import Atoms
        from rgpot.ase import RgpotRpcCalculator

        atoms = Atoms(
            numbers=[1, 1],
            positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            cell=[20.0, 20.0, 20.0],
            pbc=False,
        )

        calc = RgpotRpcCalculator.spawn(os.environ["RGPOT_POTSERV_BIN"], potential="LJ")
        atoms.calc = calc
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        calc.close()
        calc.close()

        with RgpotRpcCalculator.spawn(os.environ["RGPOT_POTSERV_BIN"], potential="LJ") as managed:
            atoms.calc = managed
            _ = atoms.get_potential_energy()
        """,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined


@pytest.mark.rpc
def test_rpc_unclosed_calculator_warns_without_abort() -> None:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    env = os.environ.copy()
    env["RGPOT_POTSERV_BIN"] = binary
    proc = _run_python(
        """
        import gc
        import os
        import warnings

        from rgpot.ase import RgpotRpcCalculator

        warnings.simplefilter("always", ResourceWarning)
        with warnings.catch_warnings(record=True) as caught:
            calc = RgpotRpcCalculator.spawn(os.environ["RGPOT_POTSERV_BIN"], potential="LJ")
            del calc
            gc.collect()
            gc.collect()
        assert any(issubclass(item.category, ResourceWarning) for item in caught), caught
        """,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined


def test_rpc_failed_init_gc_has_no_exception_ignored_stderr() -> None:
    proc = _run_python(
        """
        import gc

        import rgpot.ase.rpc as rpc_mod

        def _boom():
            raise RuntimeError("boom")

        rpc_mod._capnp = _boom
        try:
            rpc_mod.RgpotRpcCalculator()
        except RuntimeError:
            pass
        gc.collect()
        gc.collect()
        """
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
    assert "Exception ignored in:" not in proc.stderr


@pytest.mark.rpc
def test_rpc_close_when_loop_is_already_closed_releases_handles() -> None:
    binary = os.environ.get("RGPOT_POTSERV_BIN")
    if not binary or not Path(binary).is_file():
        pytest.skip("RGPOT_POTSERV_BIN is not set")
    env = os.environ.copy()
    env["RGPOT_POTSERV_BIN"] = binary
    proc = _run_python(
        """
        import os

        from rgpot.ase import RgpotRpcCalculator

        calc = RgpotRpcCalculator.spawn(os.environ["RGPOT_POTSERV_BIN"], potential="LJ")
        calc._loop.close()
        try:
            calc.close()
            raise AssertionError("close() should report pre-closed loop")
        except RuntimeError as exc:
            assert "already closed" in str(exc)
        assert calc._pot is None
        assert calc._client is None
        assert calc._connection is None
        """,
        env=env,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, combined
