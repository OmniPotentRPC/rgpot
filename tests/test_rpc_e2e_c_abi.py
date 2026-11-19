#!/usr/bin/env python3
"""End-to-end potserv RPC: configure + calculate for NWChem and CPMD with fake engines.

Requires env:
  RGPOT_POTSERV      path to potserv binary
  RGPOT_NWCHEM_ENGINE / NWCHEMC_LIBRARY  path to libnwchemc_fake_engine.so (or real)
  RGPOT_CPMD_ENGINE  / CPMDC_LIBRARY     path to libcpmdc_fake_engine.so (or real)

Run via pixi rpctest; fails hard if env is incomplete (no skip).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import capnp
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
def _load_schema() -> capnp.lib.capnp._StructModule:
    parser = capnp.SchemaParser()
    potserv = os.environ.get("RGPOT_POTSERV", "").strip()
    if potserv:
        cand = Path(potserv).resolve().parent.parent / "CppCore/rgpot/rpc/Potentials.capnp"
        if cand.is_file():
            return parser.load(str(cand))
    sys.path.insert(0, str(SCRIPT_DIR.parent / "CppCore" / "rgpot" / "rpc"))
    import schema_ledger
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "Potentials.capnp"
        schema_ledger.main(["apply", "--root", str(SCRIPT_DIR.parent), "--out", str(out)])
        return parser.load(str(out))


pot_capnp = _load_schema()

sys.path.insert(0, str(SCRIPT_DIR))
from cpmd_params import configure_cpmd  # noqa: E402
from nwchem_params import configure_nwchem, make_potential_config_none  # noqa: E402


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"required env {name} is unset (no skip allowed)")
    if not Path(val).exists():
        raise RuntimeError(f"required env {name}={val} does not exist")
    return val


def _pick_engine(*names: str) -> str:
    for n in names:
        v = os.environ.get(n, "").strip()
        if v and Path(v).exists():
            return v
    raise RuntimeError(
        f"need one of {names} pointing at an existing engine .so (no skip allowed)"
    )


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

        from rpc_integ import _compare_nlist, _local_neighbours

        geometries = [
            (
                "open_cluster",
                [0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0],
                [20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0],
                [False, False, False],
                2.0,
            ),
            (
                "cubic",
                [0.2, 0.2, 0.2, 1.4, 0.3, 0.4, 3.8, 0.2, 0.1],
                [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
                [True, True, True],
                2.2,
            ),
            (
                "triclinic",
                [0.2, 0.3, 0.4, 3.6, 0.5, 0.2, 1.0, 3.8, 0.7],
                [4.0, 0.2, 0.1, 0.3, 4.5, -0.2, 0.05, 0.15, 4.2],
                [True, True, True],
                3.2,
            ),
        ]
        for _gname, pos, box, pbc, cutoff in geometries:
            for full in (False, True):
                nfip = pot_capnp.ForceInput.new_message()
                nfip.pos = pos
                nfip.atmnrs = [8, 1, 1]
                nfip.box = box
                req = pot_capnp.NeighborRequest.new_message()
                req.cutoff = cutoff
                req.skin = 0.2
                req.format = 1 if full else 0
                req.cut = 1
                req.periodic = pbc
                req.shifts = True
                req.vectors = True
                req.distances = True
                req.mode = 1
                reply = (await pot.neighbours(nfip, req)).neighbors
                ref = _local_neighbours(pos, box, cutoff, 0.2, full, True, pbc)
                _compare_nlist(reply, ref)
        return energy, forces
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_nwchem_rpc_configure_and_calculate() -> None:
    potserv = _require_env("RGPOT_POTSERV")
    engine = _pick_engine("NWCHEMC_LIBRARY", "RGPOT_NWCHEMC_ENGINE", "RGPOT_NWCHEM_ENGINE")
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
    engine = _pick_engine("CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE", "RGPOT_CPMD_ENGINE")
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
