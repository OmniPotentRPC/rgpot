#!/usr/bin/env python3
"""Export one UMA AOTI .pt2 per unique Baker (z_set, charge, spin).

Walks Baker endpoints (reactant.con), applies the same charge/spin table as
gpr_optim materialize_systems (04_ch3o 0/2, 08_formyloxyethyl 0/2,
16_h2po4_anion -1/1, 20_hconh3_cation +1/1, else 0/1), and calls
export_uma_aoti.py once per unique composition. merge_mole stays on.

Example::

  python scripts/export_baker_uma_aoti.py \\
      --baker-dir /path/to/bench_data/baker/endpoints \\
      --only 16_h2po4_anion
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_uma_aoti import load_atoms, z_set_of  # noqa: E402


# Same table as gpr_optim bench/elja/workflow/materialize_systems.py
BAKER_CHARGE_SPIN = {
    "04_ch3o": (0, 2),
    "08_formyloxyethyl": (0, 2),
    "16_h2po4_anion": (-1, 1),
    "20_hconh3_cation": (1, 1),
}


def baker_charge_spin(label: str) -> tuple[int, int]:
    return BAKER_CHARGE_SPIN.get(label, (0, 1))


def discover_baker_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise SystemExit(f"--baker-dir is not a directory: {path}")
        return path
    env = os.environ.get("RGPOT_BAKER_ENDPOINTS")
    if env:
        path = Path(env)
        if path.is_dir():
            return path
    rel = Path("bench_data") / "baker" / "endpoints"
    roots = [Path.cwd(), Path(__file__).resolve().parents[1]]
    for root in roots:
        cand = (root / rel).resolve()
        if cand.is_dir():
            return cand
        for parent in [root, *root.parents]:
            for extra in (
                parent / "gpr_optim" / rel,
                parent / "TheochemUI" / "gpr_optim" / rel,
            ):
                if extra.is_dir():
                    return extra
    raise SystemExit(
        "Baker endpoints not found. Pass --baker-dir PATH "
        "pointing at bench_data/baker/endpoints."
    )


def collect_jobs(baker_dir: Path) -> list[dict]:
    seen: dict[tuple, str] = {}
    jobs: list[dict] = []
    for reactant in sorted(baker_dir.glob("*/reactant.con")):
        label = reactant.parent.name
        atoms = load_atoms(reactant)
        charge, spin = baker_charge_spin(label)
        key = (tuple(z_set_of(atoms)), int(charge), int(spin))
        if key in seen:
            print(
                f"skip {label}: same (z_set, charge, spin) as {seen[key]}",
                flush=True,
            )
            continue
        seen[key] = label
        jobs.append(
            {
                "label": label,
                "atoms": reactant,
                "charge": int(charge),
                "spin": int(spin),
                "z_set": list(key[0]),
            }
        )
    return jobs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baker-dir", default=None)
    p.add_argument("--out-dir", default="bench_data/uma")
    p.add_argument("--model", default="uma-s-1p1")
    p.add_argument("--task", default="omol")
    p.add_argument("--device", default="cpu")
    p.add_argument("--only", default=None, help="Export this Baker label only")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--eager-only", action="store_true")
    p.add_argument("--skip-aoti", action="store_true")
    # A static-shape package freezes the traced edge count, so a band
    # that stretches past it aborts. The sidecar makes the intramolecular
    # graph complete and the edge count constant at n(n-1).
    p.add_argument("--molecular-box", type=float, default=0.0)
    args = p.parse_args()

    baker_dir = discover_baker_dir(args.baker_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs(baker_dir)
    if args.only:
        jobs = [j for j in jobs if j["label"] == args.only]
        if not jobs:
            raise SystemExit(f"--only {args.only} is not a unique Baker composition")

    print(f"baker_dir={baker_dir} n_jobs={len(jobs)}", flush=True)
    for job in jobs:
        print(
            f"  {job['label']} z_set={job['z_set']} "
            f"Q={job['charge']} S={job['spin']}",
            flush=True,
        )
    if args.dry_run:
        return 0

    exporter = Path(__file__).resolve().parent / "export_uma_aoti.py"
    rc = 0
    for job in jobs:
        out = out_dir / f"{args.model}-{args.task}-{job['label']}.pt2"
        if args.skip_existing and out.is_file():
            print(f"exists {out}", flush=True)
            continue
        cmd = [
            sys.executable,
            str(exporter),
            "--atoms",
            str(job["atoms"]),
            "--charge",
            str(job["charge"]),
            "--spin",
            str(job["spin"]),
            "--label",
            job["label"],
            "--task",
            args.task,
            "--model",
            args.model,
            "--device",
            args.device,
            "--out",
            str(out),
        ]
        if args.eager_only:
            cmd.append("--eager-only")
        if args.skip_aoti:
            cmd.append("--skip-aoti")
        if args.molecular_box and args.molecular_box > 0.0:
            cmd += ["--molecular-box", str(args.molecular_box)]
        print("RUN", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            print(f"FAIL {job['label']} rc={proc.returncode}", file=sys.stderr)
            rc = proc.returncode
            break
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
