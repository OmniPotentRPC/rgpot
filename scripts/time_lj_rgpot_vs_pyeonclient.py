#!/usr/bin/env python3
"""Time in-process rgpot LJPot vs pyeonclient Matter on one LJ fixture.

Required race (rg.terra, repo defaults, no invented cheaper pot settings):

  A  rgpot LJPot operator() (C++ time_lj_force, else Python rgpot.LJPot)
  B  pyeonclient Matter with PotType.LJ (in-process, not potserv RPC)
  C  optional ExprPot(\"lj\") when the C++ helper was built -Dwith_expr=true

Fixture: ExprPotTest two-atom Ar (1.5 A, Z=18, 40 A box), default LJConfig.

Does not time XcKernel. XcKernel is not a Potential. Does not seed numbers.

Prints parseable KEY=value lines including A_ns_per_call and B_ns_per_call.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

TERRA_HOSTS = {"rgam5terra", "rg.terra"}
ROOT = Path(__file__).resolve().parents[1]

# ExprPotTest identity fixture (Angstrom, eV).
POSITIONS = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
NUMBERS = [18, 18]
BOX = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]]


def _require_terra() -> None:
    host = socket.gethostname().split(".")[0]
    if host not in TERRA_HOSTS and os.environ.get("TIME_LJ_ALLOW") != "1":
        sys.exit(
            f"time_lj_rgpot_vs_pyeonclient.py runs on rg.terra only (got {host})"
        )


def _parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            out[key] = val.strip()
    return out


def _find_cpp_bin(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() and os.access(p, os.X_OK) else None
    env = os.environ.get("RGPOT_TIME_LJ_BIN")
    if env:
        p = Path(env)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    for cand in ROOT.glob("**/time_lj_force"):
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    return None


def time_a_python(n_calls: int, warmup: int) -> dict[str, str]:
    import numpy as np
    import rgpot

    pos = np.array(POSITIONS, dtype=np.float64)
    types = np.array(NUMBERS, dtype=np.int32)
    box = np.array(BOX, dtype=np.float64)
    pot = rgpot.LJPot()
    energy = 0.0
    for _ in range(warmup):
        energy, forces, _var = pot(pos, types, box)
        del forces
    t0 = time.perf_counter_ns()
    for _ in range(n_calls):
        energy, forces, _var = pot(pos, types, box)
        del forces
    t1 = time.perf_counter_ns()
    ns = (t1 - t0) / float(n_calls)
    return {
        "A_ns_per_call": f"{ns:.4f}",
        "A_operator_ns_per_call": f"{ns:.4f}",
        "A_path": "rgpot.LJPot.__call__",
        "A_energy_eV": f"{float(energy):.16g}",
        "A_rgpot_file": str(getattr(rgpot, "__file__", "")),
        "A_rgpot_version": str(getattr(rgpot, "__version__", "")),
    }


def time_a_cpp(bin_path: Path, n_calls: int, warmup: int) -> dict[str, str]:
    proc = subprocess.run(
        [str(bin_path), "--n", str(n_calls), "--warmup", str(warmup)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(
            f"time_lj_force failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
        )
    kv = _parse_kv(proc.stdout)
    if "A_operator_ns_per_call" not in kv or not kv["A_operator_ns_per_call"]:
        sys.exit(f"time_lj_force missing A_operator_ns_per_call:\n{proc.stdout}")
    kv["A_ns_per_call"] = kv["A_operator_ns_per_call"]
    kv["A_path"] = f"time_lj_force:{bin_path}"
    kv["A_cpp_stdout"] = "ok"
    return kv


def time_b_pyeonclient(n_calls: int, warmup: int) -> dict[str, str]:
    import numpy as np
    import pyeonclient as pyec

    params = pyec.Parameters()
    params.potential = pyec.PotType.LJ
    if hasattr(params, "job") and hasattr(pyec, "JobType"):
        # Force eval only; Minimization is the Matter unit-test job default.
        job = getattr(pyec.JobType, "Point", None) or getattr(
            pyec.JobType, "Minimization", None
        )
        if job is not None:
            params.job = job
    if hasattr(params, "quiet"):
        params.quiet = True
    if hasattr(params, "write_log"):
        params.write_log = False
    # Same pair kernel; skip the Matter net-force reduction so B is the
    # force call, not an extra host reduction on top of LJ.
    if hasattr(params, "remove_net_force"):
        params.remove_net_force = False

    pot = pyec.make_potential(pyec.PotType.LJ, params)
    matter = pyec.Matter(pot, params)
    matter.resize(2)
    cell = np.array(BOX, dtype=np.float64)
    pos = np.array(POSITIONS, dtype=np.float64)
    matter.cell = cell
    matter.positions = pos
    if hasattr(matter, "atomic_numbers"):
        matter.atomic_numbers = np.array(NUMBERS, dtype=np.int64)
    if hasattr(matter, "masses"):
        matter.masses = np.array([39.948, 39.948], dtype=np.float64)
    if hasattr(matter, "fixed"):
        matter.fixed = np.array([0, 0], dtype=np.int64)
    if hasattr(matter, "periodic"):
        matter.periodic = True

    energy = 0.0
    # Re-assign positions every iteration so Matter's force cache misses.
    # Timed path is forces only (one computePotential); energy is read after.
    for _ in range(warmup):
        matter.positions = pos
        _ = matter.forces
    t0 = time.perf_counter_ns()
    for _ in range(n_calls):
        matter.positions = pos
        _ = matter.forces
    t1 = time.perf_counter_ns()
    energy = float(matter.potential_energy)
    ns = (t1 - t0) / float(n_calls)

    built = ""
    if hasattr(pyec, "built_with_rgpot"):
        try:
            built = str(bool(pyec.built_with_rgpot()))
        except TypeError:
            built = str(bool(pyec.built_with_rgpot))

    return {
        "B_ns_per_call": f"{ns:.4f}",
        "B_path": "pyeonclient.Matter.forces",
        "B_energy_eV": f"{energy:.16g}",
        "B_pyeonclient_file": str(getattr(pyec, "__file__", "")),
        "B_pyeonclient_version": str(getattr(pyec, "__version__", "")),
        "B_built_with_rgpot": built,
        "B_rpc": "no",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100000, help="timed force calls")
    ap.add_argument("--warmup", type=int, default=1000, help="untimed warmup calls")
    ap.add_argument(
        "--a-bin",
        type=Path,
        default=None,
        help="time_lj_force binary (else RGPOT_TIME_LJ_BIN or search)",
    )
    ap.add_argument(
        "--a-python",
        action="store_true",
        help="force A through rgpot.LJPot even if a C++ binary exists",
    )
    args = ap.parse_args()
    if args.n <= 0 or args.warmup <= 0:
        sys.exit("--n and --warmup must be positive")

    _require_terra()

    print(f"host={socket.gethostname()}")
    print("fixture=exprpot_two_atom")
    print("n_atoms=2")
    print(f"n_calls={args.n}")
    print(f"warmup={args.warmup}")
    print("pot=LJ default LJConfig (u0=1 eV, cutoff=15 A, psi=1 A)")
    print("note=XcKernel is not a Potential and is not timed")

    a_bin = None if args.a_python else _find_cpp_bin(
        str(args.a_bin) if args.a_bin else None
    )
    if a_bin is not None:
        a = time_a_cpp(a_bin, args.n, args.warmup)
    else:
        try:
            a = time_a_python(args.n, args.warmup)
        except ImportError as exc:
            sys.exit(
                "A needs time_lj_force or importable rgpot "
                f"({exc}). Build: meson setup -Dwith_examples=true "
                "-Dwith_expr=true --buildtype=release && meson compile "
                "-C <build> time_lj_force"
            )

    try:
        b = time_b_pyeonclient(args.n, args.warmup)
    except ImportError as exc:
        sys.exit(
            "B needs importable pyeonclient "
            f"({exc}). In-process Matter only; do not point this at potserv."
        )

    merged = dict(a)
    merged.update(b)
    # Re-print A/B keys in a stable order after the preamble.
    for key in (
        "A_ns_per_call",
        "A_operator_ns_per_call",
        "A_handle_ns_per_call",
        "A_handle_note",
        "A_path",
        "A_energy_eV",
        "A_rgpot_file",
        "A_rgpot_version",
        "B_ns_per_call",
        "B_path",
        "B_energy_eV",
        "B_pyeonclient_file",
        "B_pyeonclient_version",
        "B_built_with_rgpot",
        "B_rpc",
        "C_expr_ns_per_call",
        "C_expr_note",
        "C_expr_energy_eV",
        "C_over_A",
    ):
        if key in merged and merged[key] != "":
            print(f"{key}={merged[key]}")

    a_ns = float(merged["A_ns_per_call"])
    b_ns = float(merged["B_ns_per_call"])
    if a_ns > 0.0:
        print(f"B_over_A={b_ns / a_ns:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
