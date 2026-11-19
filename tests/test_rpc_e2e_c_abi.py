#!/usr/bin/env python3
"""End to end potserv RPC: configure and calculate for NWChem and CPMD.

Needs:
  RGPOT_POTSERV      path to the potserv binary
  RGPOT_NWCHEM_ENGINE / NWCHEMC_LIBRARY  path to libnwchemc (fake or real)
  RGPOT_CPMD_ENGINE  / CPMDC_LIBRARY     path to libcpmdc (fake or real)

Skips when potserv or the named engine library is unset. A variable that
is set to a missing path fails. capnp is imported only when a test runs.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SCRIPT_DIR / "../CppCore/rgpot/rpc/Potentials.capnp"
sys.path.insert(0, str(SCRIPT_DIR))

capnp = None
pot_capnp = None
configure_nwchem = None
configure_cpmd = None
make_potential_config_none = None


def test_potserv_unset_skips(monkeypatch) -> None:
    monkeypatch.delenv("RGPOT_POTSERV", raising=False)
    with pytest.raises(pytest.skip.Exception, match="unset"):
        _require_env("RGPOT_POTSERV")


def test_potserv_set_but_missing_fails(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-potserv"
    monkeypatch.setenv("RGPOT_POTSERV", str(missing))
    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _require_env("RGPOT_POTSERV")


def test_engine_unset_skips(monkeypatch) -> None:
    for name in ("NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(pytest.skip.Exception, match="unset"):
        _pick_engine("NWChem", "NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE")


def test_engine_set_but_missing_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RGPOT_NWCHEMC_ENGINE", raising=False)
    monkeypatch.delenv("RGPOT_NWCHEM_ENGINE", raising=False)
    monkeypatch.setenv("NWCHEMC_LIBRARY", str(tmp_path / "missing-engine.so"))
    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _pick_engine("NWChem", "NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE")


def test_capnp_is_not_imported_at_module_level() -> None:
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert "capnp" not in names
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "capnp"


def _prepare() -> None:
    """Import capnp and the schema helpers on the first test that runs."""
    global capnp, pot_capnp, configure_nwchem, configure_cpmd, make_potential_config_none
    if capnp is not None:
        return
    try:
        import capnp as capnp_mod
    except ImportError as exc:
        pytest.fail(f"pycapnp is required when potserv tests run: {exc}")
    from cpmd_params import configure_cpmd as _configure_cpmd
    from nwchem_params import configure_nwchem as _configure_nwchem
    from nwchem_params import make_potential_config_none as _none

    capnp = capnp_mod
    pot_capnp = capnp.load(str(SCHEMA_PATH))
    configure_nwchem = _configure_nwchem
    configure_cpmd = _configure_cpmd
    make_potential_config_none = _none


def _require_env(name: str) -> str:
    if name not in os.environ:
        pytest.skip(f"{name} is unset")
    raw = os.environ[name].strip()
    if not raw or not Path(raw).is_file():
        pytest.fail(f"{name} is set to {raw!r}, which is not a file")
    return raw


def _pick_engine(kind: str, *names: str) -> str:
    saw = False
    for name in names:
        if name not in os.environ:
            continue
        saw = True
        raw = os.environ[name].strip()
        if raw and Path(raw).is_file():
            return raw
        pytest.fail(f"{name} is set to {raw!r}, which is not a file")
    joined = ", ".join(names)
    if not saw:
        pytest.skip(f"optional {kind} engine library is unset ({joined})")
    pytest.fail(f"optional {kind} engine library is not a file ({joined})")


async def _connect(port: int, retries: int = 20):
    for _ in range(retries):
        try:
            return await capnp.AsyncIoStream.create_connection(
                host="127.0.0.1", port=port
            )
        except OSError:
            await asyncio.sleep(0.25)
    raise RuntimeError(f"failed to connect to potserv on port {port}")


async def _water_force_input():
    fip = pot_capnp.ForceInput.new_message()
    # O-H-H water-ish (angstrom)
    pos = [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0]
    fip.init("pos", len(pos))
    for i, p in enumerate(pos):
        fip.pos[i] = p
    fip.init("atmnrs", 3)
    fip.atmnrs[0] = 8
    fip.atmnrs[1] = 1
    fip.atmnrs[2] = 1
    box = [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0]
    fip.init("box", len(box))
    for i, b in enumerate(box):
        fip.box[i] = b
    fip.lengthUnit = "angstrom"
    fip.energyUnit = "eV"
    return fip


async def _run_backend(
    potserv: str,
    port: int,
    pot_name: str,
    engine_env: dict[str, str],
    configure_fn,
):
    env = os.environ.copy()
    env.update(engine_env)
    proc = subprocess.Popen(
        [potserv, str(port), pot_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        connection = await _connect(port)
        client = capnp.TwoPartyClient(connection)
        pot = client.bootstrap().cast_as(pot_capnp.Potential)

        r0 = await pot.configure(make_potential_config_none(pot_capnp))
        assert r0.ok, f"configure(none) failed: {r0.message}"

        ok, msg = await configure_fn(pot)
        assert ok, f"configure({pot_name}) failed: {msg}"

        fip = await _water_force_input()
        result = await pot.calculate(fip)
        energy = float(result.result.energy)
        forces = list(result.result.forces)
        assert np.isfinite(energy), f"non-finite energy: {energy}"
        assert energy != 0.0, "energy is zero (engine likely not loaded)"
        assert len(forces) == 9, f"expected 9 force components, got {len(forces)}"
        assert all(np.isfinite(f) for f in forces), f"non-finite forces: {forces}"
        # At least one non-zero force from fake engines
        assert any(abs(f) > 1e-12 for f in forces), f"all-zero forces: {forces}"
        return energy, forces
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_nwchem_rpc_configure_and_calculate() -> None:
    potserv = _require_env("RGPOT_POTSERV")
    engine = _pick_engine(
        "NWChem", "NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE"
    )
    _prepare()
    port = int(os.environ.get("RGPOT_E2E_NWCHEM_PORT", "19101"))

    async def cfg(pot):
        return await configure_nwchem(
            pot,
            pot_capnp,
            basis="sto-3g",
            theory="scf",
            scf_type="rhf",
            engine_path=engine,
        )

    energy, forces = asyncio.run(
        capnp.run(
            _run_backend(
                potserv,
                port,
                "NWChem",
                {
                    "NWCHEMC_LIBRARY": engine,
                    "RGPOT_NWCHEMC_ENGINE": engine,
                    "RGPOT_NWCHEM_ENGINE": engine,
                },
                cfg,
            )
        )
    )
    # Fake nwchemc returns 0.25 Ha -> eV conversion in frontend
    assert energy > 0.0
    print(f"NWChem E2E energy={energy} forces0={forces[0]}")


def test_cpmd_rpc_configure_and_calculate() -> None:
    potserv = _require_env("RGPOT_POTSERV")
    engine = _pick_engine(
        "CPMD", "CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE", "RGPOT_CPMD_ENGINE"
    )
    _prepare()
    port = int(os.environ.get("RGPOT_E2E_CPMD_PORT", "19102"))

    async def cfg(pot):
        return await configure_cpmd(
            pot,
            pot_capnp,
            functional="BLYP",
            task="gradient",
            engine_path=engine,
        )

    energy, forces = asyncio.run(
        capnp.run(
            _run_backend(
                potserv,
                port,
                "CPMD",
                {
                    "CPMDC_LIBRARY": engine,
                    "RGPOT_CPMDC_ENGINE": engine,
                    "RGPOT_CPMD_ENGINE": engine,
                },
                cfg,
            )
        )
    )
    assert energy > 0.0
    print(f"CPMD E2E energy={energy} forces0={forces[0]}")


if __name__ == "__main__":
    test_nwchem_rpc_configure_and_calculate()
    test_cpmd_rpc_configure_and_calculate()
    print("test_rpc_e2e_c_abi: all ok")
