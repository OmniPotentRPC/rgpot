#!/usr/bin/env python3
"""Evaluate the named water geometry through potserv and a CPMD engine.

The geometry is tests/data/cpmd/water.xyz. The cell is a cube of side
20 angstrom. The in-tree stand-in engine returns 0.77 for that cell.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import subprocess
import sys
from pathlib import Path

import capnp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WATER_XYZ = SCRIPT_DIR / "data" / "cpmd" / "water.xyz"
SCHEMA_PATH = REPO_ROOT / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
CELL_ANGSTROM = 20.0
SYMBOL_Z = {"O": 8, "H": 1}

sys.path.insert(0, str(SCRIPT_DIR))
from cpmd_params import configure_cpmd  # noqa: E402
from nwchem_params import make_potential_config_none  # noqa: E402

pot_capnp = capnp.load(str(SCHEMA_PATH))


def load_water(path: Path):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 5:
        raise SystemExit(f"{path} must be a 3-atom xyz with a title")
    count = int(lines[0])
    name = lines[1]
    if count != 3 or name != "water":
        raise SystemExit(f"{path} must be the named water geometry")
    pos = []
    atomic_numbers = []
    for line in lines[2:]:
        symbol, x, y, z = line.split()
        if symbol not in SYMBOL_Z:
            raise SystemExit(f"{path} has an unexpected element {symbol}")
        pos.extend((float(x), float(y), float(z)))
        atomic_numbers.append(SYMBOL_Z[symbol])
    if atomic_numbers != [8, 1, 1]:
        raise SystemExit(f"{path} must list oxygen, then two hydrogens")
    return name, pos, atomic_numbers


def engine_path(explicit: str) -> Path:
    chosen = explicit.strip()
    if not chosen:
        for key in ("RGPOT_CPMD_ENGINE", "CPMDC_LIBRARY", "RGPOT_CPMDC_ENGINE"):
            chosen = os.environ.get(key, "").strip()
            if chosen:
                break
    if not chosen:
        chosen = str(REPO_ROOT / "bbdir" / "CppCore" / "libcpmdc_fake_engine.so")
    path = Path(chosen)
    if not path.is_file():
        raise SystemExit(f"CPMD engine not found: {path}")
    return path


async def _connect(port: int):
    for _ in range(20):
        try:
            return await capnp.AsyncIoStream.create_connection(
                host="127.0.0.1", port=port
            )
        except OSError:
            await asyncio.sleep(0.25)
    raise RuntimeError(f"failed to connect to potserv on port {port}")


async def evaluate(potserv: str, port: int, engine: Path):
    env = os.environ.copy()
    engine_s = str(engine)
    env["RGPOT_CPMD_ENGINE"] = engine_s
    env["CPMDC_LIBRARY"] = engine_s
    env["RGPOT_CPMDC_ENGINE"] = engine_s
    proc = subprocess.Popen(
        [potserv, str(port), "CPMD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        connection = await _connect(port)
        client = capnp.TwoPartyClient(connection)
        pot = client.bootstrap().cast_as(pot_capnp.Potential)

        none = await pot.configure(make_potential_config_none(pot_capnp))
        if not none.ok:
            raise RuntimeError(f"configure(none) failed: {none.message}")
        ok, message = await configure_cpmd(
            pot,
            pot_capnp,
            functional="BLYP",
            task="gradient",
            engine_path=engine_s,
        )
        if not ok:
            raise RuntimeError(f"configure(CPMD) failed: {message}")

        _name, pos, atomic_numbers = load_water(WATER_XYZ)
        force = pot_capnp.ForceInput.new_message()
        force.init("pos", len(pos))
        for i, value in enumerate(pos):
            force.pos[i] = value
        force.init("atmnrs", len(atomic_numbers))
        for i, value in enumerate(atomic_numbers):
            force.atmnrs[i] = value
        box = [
            CELL_ANGSTROM, 0.0, 0.0,
            0.0, CELL_ANGSTROM, 0.0,
            0.0, 0.0, CELL_ANGSTROM,
        ]
        force.init("box", len(box))
        for i, value in enumerate(box):
            force.box[i] = value
        force.lengthUnit = "angstrom"
        force.energyUnit = "eV"

        result = await pot.calculate(force)
        return float(result.result.energy)
    finally:
        proc.kill()
        proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", required=True, help="Path to potserv")
    parser.add_argument("--engine", default="", help="Path to libcpmdc.so")
    parser.add_argument("--port", type=int, default=19111)
    parser.add_argument(
        "--expect",
        type=float,
        default=None,
        help="Require the printed energy to match this value",
    )
    args = parser.parse_args()

    server = Path(args.server_bin)
    if not server.is_file():
        print(f"potserv not found: {server}", file=sys.stderr)
        return 1
    engine = engine_path(args.engine)
    name, _pos, _atomic_numbers = load_water(WATER_XYZ)
    try:
        energy = asyncio.run(
            capnp.run(evaluate(str(server), args.port, engine))
        )
    except Exception as exc:
        print(f"water evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not math.isfinite(energy):
        print(f"water energy is not finite: {energy}", file=sys.stderr)
        return 1
    print(f"water geometry: {name}")
    print(f"water energy: {energy:.12g}")
    if args.expect is not None and not math.isclose(
        energy, args.expect, rel_tol=0.0, abs_tol=1e-12
    ):
        print(
            f"water energy {energy:.12g} != {args.expect:.12g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
