#!/usr/bin/env python3
"""Write synth.eam.fs, a two-element eam/fs table for the EAMTable tests.

The table is not a fitted model. It is small (nrho = nr = 8) so that test
cells can reach both ends of the grids: the cutoff lies past the last r
knot (nr - 1) * dr, and rhomax = (nrho - 1) * drho is below the density of
a compressed atom. Every ordered pair of elements has its own density.

  F_a(rho)   = -E_a sqrt(rho) + G_a rho^2
  rho_ab(r)  = A_ab exp(-B_ab (r - 2.5)), from an a atom onto a b atom
  phi_ab(r)  = D_ab (exp(-2 K (r - R_ab)) - 2 exp(-K (r - R_ab)))

Usage: python3 synth_fs.py [OUT], OUT defaults to synth.eam.fs.
"""
import math
import sys

NRHO, DRHO = 8, 0.6
NR, DR = 8, 0.7
CUT = 5.3
ELEMENTS = [("Cu", 29, 63.546, 3.615), ("Ag", 47, 107.8682, 4.09)]
E = [2.0, 2.6]
G = [0.05, 0.03]
A = [[1.0, 0.6], [1.4, 0.8]]
B = [[1.0, 0.9], [1.1, 0.8]]
D = [[0.4, 0.3], [0.3, 0.5]]
R = [[2.55, 2.75], [2.75, 2.9]]
K = 1.2


def rows(values):
    return ["  ".join(repr(v) for v in values[k:k + 5]) for k in range(0, len(values), 5)]


def main(out):
    rho = [k * DRHO for k in range(NRHO)]
    r = [k * DR for k in range(NR)]
    lines = [
        "Synthetic eam/fs table for the rgpot EAMTable tests, not a fitted model",
        f"Written by synth_fs.py: nrho = nr = {NRHO}, drho = {DRHO}, dr = {DR}, cutoff = {CUT}",
        "F = -E sqrt(rho) + G rho^2, rho = A exp(-B (r - 2.5)), phi = Morse",
        f"{len(ELEMENTS)} " + " ".join(e[0] for e in ELEMENTS),
        f"{NRHO} {DRHO!r} {NR} {DR!r} {CUT!r}",
    ]
    for a, (_, z, mass, lat) in enumerate(ELEMENTS):
        lines.append(f"{z} {mass!r} {lat!r} fcc")
        lines += rows([-E[a] * math.sqrt(x) + G[a] * x * x for x in rho])
        for b in range(len(ELEMENTS)):
            lines += rows([A[a][b] * math.exp(-B[a][b] * (x - 2.5)) for x in r])
    for a in range(len(ELEMENTS)):
        for b in range(a + 1):
            phi = [D[a][b] * (math.exp(-2 * K * (x - R[a][b])) - 2 * math.exp(-K * (x - R[a][b])))
                   for x in r]
            lines += rows([x * p for x, p in zip(r, phi)])
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synth.eam.fs")
