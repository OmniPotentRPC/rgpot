# Copyright 2023--present rgpot developers
"""AOTI package that reproduces the Claisen static-edge contract.

The graph is traced at n(n-1) edges. Energy is a sum of a polynomial
envelope that is zero, with zero derivative, at model_cutoff. Metadata
records the old widened neighbor cutoff (10 angstrom) separately from
model_cutoff, which is what a package exported with --molecular-box 25
used to embed. UmaPot on that package used to drop a pair past 10
angstrom and miss the compiled edge count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from uma_edges import complete_ordered_edges  # noqa: E402


def envelope_reference(pos, cutoff: float):
    """Same sum UmaPot's probe graph and the C++ reference use."""
    n = len(pos)
    c2 = cutoff * cutoff
    energy = 0.0
    forces = [[0.0, 0.0, 0.0] for _ in range(n)]
    for center in range(n):
        for neighbor in range(n):
            if neighbor == center:
                continue
            d = [pos[neighbor][a] - pos[center][a] for a in range(3)]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if not (r2 < c2):
                continue
            x = r2 / c2
            energy += (1.0 - x) * (1.0 - x)
            scale = (-4.0 / c2) * (1.0 - x)
            for a in range(3):
                g = scale * d[a]
                forces[neighbor][a] -= g
                forces[center][a] += g
    return energy, forces


class EnvelopeProbe(nn.Module):
    def __init__(self, cutoff: float, nedges: int):
        super().__init__()
        self.register_buffer(
            "cutoff", torch.tensor(float(cutoff), dtype=torch.float64)
        )
        self.register_buffer("slots", torch.arange(int(nedges), dtype=torch.long))

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
        del atomic_numbers, pbc, charge, spin, batch, natoms
        ei = edge_index.index_select(1, self.slots)
        off = cell_offsets.index_select(0, self.slots)
        neighbor = ei[0]
        center = ei[1]
        dvec = pos[neighbor] - pos[center] + off @ cell[0]
        r2 = (dvec * dvec).sum(-1)
        c2 = self.cutoff * self.cutoff
        inside = r2 < c2
        x = r2 / c2
        zero = torch.zeros_like(r2)
        energy = torch.where(inside, (1.0 - x) ** 2, zero).sum()
        scale = torch.where(inside, (-4.0 / c2) * (1.0 - x), zero)
        g = dvec * scale.unsqueeze(-1)
        forces = torch.zeros_like(pos)
        forces = forces.index_add(0, neighbor, -g)
        forces = forces.index_add(0, center, g)
        return energy, forces


def _example(nedges: int, natoms: int, model_cutoff: float):
    # A compact chain plus one pair pulled out past both cutoffs, so the
    # trace sees a full ordered list that includes a non-contributing pair.
    pos = []
    for i in range(natoms):
        pos.append((0.2 * i, 0.05 * ((i * 3) % 5), 0.0))
    pos[5] = (-4.0, -2.0, 0.4)
    pos[11] = (4.5, 3.2, -0.8)
    neighbor, center, shifts = complete_ordered_edges(pos)
    if len(neighbor) != nedges:
        raise SystemExit(f"trace edges {len(neighbor)} != {nedges}")
    z = [6, 6, 6, 6, 6, 8, 1, 1, 1, 1, 1, 1, 1, 1]
    dtype = torch.float64
    pos_t = torch.tensor(pos, dtype=dtype)
    edge_index = torch.tensor([neighbor, center], dtype=torch.long)
    offsets = torch.tensor(shifts, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0) * 25.0
    pbc = torch.tensor([[True, True, True]])
    atomic_numbers = torch.tensor(z, dtype=torch.long)
    charge = torch.tensor([0], dtype=torch.long)
    spin = torch.tensor([1], dtype=torch.long)
    batch = torch.zeros(natoms, dtype=torch.long)
    natoms_t = torch.tensor([natoms], dtype=torch.long)
    example = (
        pos_t,
        atomic_numbers,
        cell,
        pbc,
        edge_index,
        offsets,
        charge,
        spin,
        batch,
        natoms_t,
    )
    ref_e, ref_f = envelope_reference(pos, model_cutoff)
    return example, ref_e, ref_f


def _compile(exported, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fn = None
    last = None
    for spec in (
        "torch.export.aoti_compile_and_package",
        "torch._inductor.aoti_compile_and_package",
    ):
        mod, name = spec.rsplit(".", 1)
        try:
            module = __import__(mod, fromlist=[name])
            fn = getattr(module, name)
            break
        except Exception as exc:
            last = exc
    if fn is None:
        raise RuntimeError("aoti_compile_and_package is missing") from last
    fn(
        exported,
        package_path=str(path),
        inductor_configs={
            "aot_inductor.metadata": {str(k): str(v) for k, v in metadata.items()}
        },
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--natoms", type=int, default=14)
    p.add_argument("--model-cutoff", type=float, default=6.0)
    p.add_argument("--graph-cutoff", type=float, default=10.0)
    p.add_argument("--molecular-box", type=float, default=25.0)
    args = p.parse_args()
    if args.natoms != 14:
        raise SystemExit("this probe is the 14-atom Claisen composition")
    nedges = args.natoms * (args.natoms - 1)
    example, ref_e, ref_f = _example(nedges, args.natoms, args.model_cutoff)
    module = EnvelopeProbe(args.model_cutoff, nedges).eval()
    with torch.no_grad():
        energy, forces = module(*example)
    got_f = forces.detach().cpu().tolist()
    if abs(float(energy) - ref_e) > 1e-10:
        raise SystemExit(f"eager energy {float(energy)} != {ref_e}")
    for i in range(args.natoms):
        for a in range(3):
            if abs(got_f[i][a] - ref_f[i][a]) > 1e-8:
                raise SystemExit(f"eager force[{i}][{a}] mismatch")
    exported = torch.export.export(module, example, strict=False)
    counts = {"1": 8, "6": 5, "8": 1}
    meta = {
        "cutoff": float(args.graph_cutoff),
        "model_cutoff": float(args.model_cutoff),
        "molecular_box": float(args.molecular_box),
        "max_neighbors": 300,
        "batch_max": 0,
        "task_name": "omol",
        "charge": 0,
        "spin": 1,
        "z_set": [1, 6, 8],
        "natoms": args.natoms,
        "counts": json.dumps({str(z): n for z, n in sorted(counts.items())}),
        "nedges_min": nedges,
        "nedges_max": nedges,
        "pos_dtype": "float64",
        "label": "claisen-cutoff-probe",
        "edge_convention": "fairchem_neighbor_center",
    }
    _compile(exported, args.out, meta)
    print(f"WROTE {args.out.resolve()} nedges={nedges} E={ref_e:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
