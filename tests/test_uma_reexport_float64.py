#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""A pre-check package is refused, and the float64 re-export loads.

The fixture is a one-parameter module. Its package omits the edge
range and both dtype keys, which is the shape of a package exported
before those checks. UmaPot names that refusal. The same weights and
the original export config are re-exported, and the new package loads
with both dtype keys equal to float64. A 1e-7 angstrom shift survives,
so the graph was traced in float64.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from export_uma_aoti import (  # noqa: E402
    aoti_package,
    apply_export_dtype,
    runtime_metadata,
)

# The three-atom line. Same coordinates as the probe and as
# LINE_POSITIONS in the exporter. Exact in float32.
LINE_NUMBERS = [6, 7, 1]
LINE_POSITIONS = [
    [12.5, 12.5, 12.5],
    [12.5, 12.5, 13.5],
    [12.5, 12.5, 11.5],
]
LINE_CELL = [[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]]


class ScaleEnergy(nn.Module):
    """Same state dict as TinyWeightedEnergy: one parameter, ``scale``."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(
        self,
        pos,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        cell_offsets,
        charge,
        spin,
        batch,
        natoms,
    ):
        energy = self.scale * pos.reshape(-1).sum()
        forces = -self.scale * torch.ones_like(pos)
        return energy, forces


def _drop_keys(path: Path, keys: tuple[str, ...]) -> None:
    blob = path.read_bytes()
    source = zipfile.ZipFile(BytesIO(blob))
    out = BytesIO()
    with zipfile.ZipFile(out, "w") as dest:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename.endswith(".json"):
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    for key in keys:
                        payload.pop(key, None)
                    data = json.dumps(payload).encode()
            dest.writestr(info, data)
    path.write_bytes(out.getvalue())


def _example(dtype):
    pos = torch.tensor(LINE_POSITIONS, dtype=dtype)
    numbers = torch.tensor(LINE_NUMBERS, dtype=torch.long)
    cell = torch.tensor(LINE_CELL, dtype=dtype).reshape(1, 3, 3)
    pbc = torch.tensor([[True, True, True]])
    neighbor, center = [], []
    n = pos.shape[0]
    for c in range(n):
        for nbr in range(n):
            if nbr != c:
                neighbor.append(nbr)
                center.append(c)
    edge_index = torch.tensor([neighbor, center], dtype=torch.long)
    nedges = edge_index.shape[1]
    offsets = torch.zeros((nedges, 3), dtype=dtype)
    charge = torch.tensor([0], dtype=torch.long)
    spin = torch.tensor([1], dtype=torch.long)
    batch = torch.zeros(n, dtype=torch.long)
    natoms = torch.tensor([n], dtype=torch.long)
    return (
        pos,
        numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms,
    )


def _write_precheck(pkg: Path, weights: Path) -> None:
    """Package in the pre-check format: no edge range and no dtype keys."""
    module = ScaleEnergy()
    module.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    module = apply_export_dtype(module.eval(), "float32")
    example = _example(torch.float32)
    exported = torch.export.export(module, example, strict=False)
    meta = runtime_metadata(
        pkg / "package.pt2",
        6.0,
        300,
        example,
        torch.float32,
        model_dtype=torch.float32,
        task_name="omol",
        charge=0,
        spin=1,
        z_set=[1, 6, 7],
        counts={1: 1, 6: 1, 7: 1},
        label="line",
        model="tiny-weighted",
    )
    for key in (
        "pos_dtype",
        "model_dtype",
        "nedges_min",
        "nedges_max",
        "edge_index_shape",
        "pos_shape",
        "molecular",
    ):
        meta.pop(key, None)
    aoti_package(exported, pkg / "package.pt2", metadata=meta)
    _drop_keys(
        pkg / "package.pt2",
        (
            "pos_dtype",
            "model_dtype",
            "nedges_min",
            "nedges_max",
            "edge_index_shape",
            "pos_shape",
            "molecular",
        ),
    )


def _probe(probe: str, package: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [probe, str(package)],
        check=False,
        capture_output=True,
        text=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: test_uma_reexport_float64.py PROBE", file=sys.stderr)
        return 2
    probe = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="uma-reexport-") as work_dir:
        return _run(probe, Path(work_dir))


def _run(probe: str, work: Path) -> int:
    pkg = work / "packages" / "line"
    pkg.mkdir(parents=True)
    module = ScaleEnergy()
    with torch.no_grad():
        module.scale.fill_(2.0)
    weights = pkg / "weights.pt"
    torch.save(module.state_dict(), weights)
    config = {
        "kind": "tiny-weighted",
        "weights": "weights.pt",
        "charge": 0,
        "spin": 1,
        "task": "omol",
        "label": "line",
        "model": "tiny-weighted",
        "cutoff": 6.0,
        "max_neighbors": 300,
        "molecular_box": 0.0,
        "batch_max": 0,
        "device": "cpu",
        "numbers": LINE_NUMBERS,
        "positions": LINE_POSITIONS,
        "cell": LINE_CELL,
    }
    (pkg / "export.json").write_text(json.dumps(config))
    _write_precheck(pkg, weights)

    refused = _probe(probe, pkg / "package.pt2")
    refused_text = refused.stdout + refused.stderr
    print(refused_text, end="" if refused_text.endswith("\n") else "\n", flush=True)
    if refused.returncode == 0 or "nedges" not in refused_text or "absent" not in refused_text:
        print("FAIL: pre-check package was not refused", file=sys.stderr)
        return 1

    out = work / "out"
    receipt = work / "receipt.txt"
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    script = _ROOT / "scripts" / "reexport_uma_float64.sh"
    proc = subprocess.run(
        [str(script), str(work / "packages"), str(out), str(receipt)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        print("FAIL: re-export script exited", proc.returncode, file=sys.stderr)
        return 1

    line = receipt.read_text().strip().splitlines()
    if len(line) != 1 or "dtype=float64" not in line[0]:
        print("FAIL: receipt", line, file=sys.stderr)
        return 1
    fields = dict(part.split("=", 1) for part in line[0].split())
    exported = out / "line.pt2"
    if fields["input_sha256"] != _sha256(pkg / "package.pt2"):
        print("FAIL: input sha256", file=sys.stderr)
        return 1
    if fields["output_sha256"] != _sha256(exported):
        print("FAIL: output sha256", file=sys.stderr)
        return 1
    if fields["input_sha256"] == fields["output_sha256"]:
        print("FAIL: re-export did not write a new package", file=sys.stderr)
        return 1
    if fields["dtype"] != "float64" or fields["package"] != "line":
        print("FAIL: receipt fields", fields, file=sys.stderr)
        return 1

    loaded = _probe(probe, exported)
    loaded_text = loaded.stdout + loaded.stderr
    print(loaded_text, end="" if loaded_text.endswith("\n") else "\n", flush=True)
    if loaded.returncode != 0:
        print("FAIL: re-exported package did not load", file=sys.stderr)
        return 1
    got = dict(
        part.split("=", 1)
        for part in loaded.stdout.splitlines()
        if "=" in part
    )
    if got.get("pos_dtype") != "float64" or got.get("model_dtype") != "float64":
        print("FAIL: dtype", got, file=sys.stderr)
        return 1
    energy = float(got["energy"])
    difference = float(got["difference"])
    if abs(energy - 225.0) > 1e-9:
        print("FAIL: energy", energy, file=sys.stderr)
        return 1
    if abs(difference - 2e-7) > 1e-12:
        print("FAIL: difference", difference, file=sys.stderr)
        return 1
    print("ok pre-check refused and float64 re-export loaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
