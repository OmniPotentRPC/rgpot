#!/usr/bin/env python3
# MIT License
# Copyright 2023--present rgpot developers
"""Independent high-precision evaluation of the Quantum Sutton-Chen potential.

This module shares no code with rgpot. It evaluates the equations of Qi,
Cagin, Kimura and Goddard, Phys. Rev. B 59, 3527 (1999), Eqs. (1a)-(1c) and
(2), in 50-digit ``decimal`` arithmetic:

    U      = sum_i [ 1/2 sum_{j != i} eps_ij (a_ij / r_ij)^n_ij
                     - c_i eps_ii sqrt(rho_i) ]
    rho_i  = sum_{j != i} (a_ij / r_ij)^m_ij

    eps_ij = sqrt(eps_ii eps_jj)       n_ij = (n_ii + n_jj) / 2
    m_ij   = (m_ii + m_jj) / 2         a_ij = (a_ii + a_jj) / 2

There is no combination rule for c_i. The quantum correction of the Q-SC
force field lives entirely in the fitted parameters; the functional form is
that of the classical Sutton-Chen potential, so no extra term exists.

Periodic cells are summed over every lattice image with r <= cutoff, which
makes the oracle independent of any minimum-image or neighbour-list logic.
Forces and the stress come from the analytic gradient; ``selfcheck`` compares
both with central differences taken in the same 50-digit arithmetic (step
1e-20, error below 1e-30), so the gradient code is itself validated.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
from decimal import Context, Decimal, localcontext

D = Decimal


def _digits():
    """A 50-digit context that belongs to this call.

    ``decimal`` contexts are thread-local. A precision set while the module
    is imported reaches only that thread, so a worker starts at the default
    of 28 and the finite-difference self check (step 1e-20, bound 1e-28)
    fails. Each public entry builds a new context. ``localcontext`` copies
    it onto the calling thread, and no context object is shared.
    """
    return localcontext(Context(prec=50))


# Z -> (n, m, epsilon [eV], c, a [Angstrom]). Same values as the rgpot default
# table; Cu a = 3.6030 is the PRB 59, 3527 Table I entry.
QSC_TABLE = {
    28: (10.0, 5.0, 0.0073767, 84.745, 3.5157),
    29: (10.0, 5.0, 0.0057921, 84.843, 3.6030),
    45: (13.0, 5.0, 0.0024612, 305.499, 3.7984),
    46: (12.0, 6.0, 0.0032864, 148.205, 3.8813),
    47: (11.0, 6.0, 0.0039450, 96.524, 4.0691),
    77: (13.0, 6.0, 0.0037674, 224.815, 3.8344),
    78: (11.0, 7.0, 0.0097894, 71.336, 3.9163),
    79: (11.0, 8.0, 0.0078052, 53.581, 4.0651),
}
SYMBOLS = {28: "Ni", 29: "Cu", 45: "Rh", 46: "Pd", 47: "Ag", 77: "Ir", 78: "Pt", 79: "Au"}
VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _dec_params(params):
    """Exact binary values of the double inputs, as Decimals."""
    return {z: tuple(D(float(x)) for x in row) for z, row in params.items()}


def _pair(pa, pb):
    n_a, m_a, e_a, _, a_a = pa
    n_b, m_b, e_b, _, a_b = pb
    return {
        "n": (n_a + n_b) / 2,
        "m": (m_a + m_b) / 2,
        "eps": (e_a * e_b).sqrt(),
        "a": (a_a + a_b) / 2,
    }


def _images(pos_i, pos_j, cell, periodic, cutoff):
    """Yield (dx, dy, dz) = r_i - r_j - shift for every image with r <= cutoff."""
    d0 = [pos_i[k] - pos_j[k] for k in range(3)]
    if not periodic:
        if d0[0] * d0[0] + d0[1] * d0[1] + d0[2] * d0[2] <= cutoff * cutoff:
            yield tuple(d0)
        return
    cf = [[float(cell[r][k]) for k in range(3)] for r in range(3)]
    inv = _inverse3(cf)
    vol = abs(_det3(cf))
    heights = []
    for r in range(3):
        o1, o2 = cf[(r + 1) % 3], cf[(r + 2) % 3]
        cr = [
            o1[1] * o2[2] - o1[2] * o2[1],
            o1[2] * o2[0] - o1[0] * o2[2],
            o1[0] * o2[1] - o1[1] * o2[0],
        ]
        heights.append(vol / math.sqrt(sum(c * c for c in cr)))
    reach = [int(math.ceil(float(cutoff) / h)) + 1 for h in heights]
    # Fractional separation d0 = sum_r f_r cell[r]; solve f from row vectors.
    f = [sum(float(d0[k]) * inv[k][r] for k in range(3)) for r in range(3)]
    centre = [round(x) for x in f]
    cut2 = cutoff * cutoff
    for n0 in range(-reach[0], reach[0] + 1):
        for n1 in range(-reach[1], reach[1] + 1):
            for n2 in range(-reach[2], reach[2] + 1):
                nn = (centre[0] + n0, centre[1] + n1, centre[2] + n2)
                d = tuple(
                    d0[k] - (nn[0] * cell[0][k] + nn[1] * cell[1][k] + nn[2] * cell[2][k])
                    for k in range(3)
                )
                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= cut2:
                    yield d


def _det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _inverse3(m):
    det = _det3(m)
    c = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
    return [[c[i][j] / det for j in range(3)] for i in range(3)]


def _density_term(rule, pair, src, r):
    """(a_ij / r)^m_ij for the paper's rule, (a_j / r)^m_j for eOn's rule."""
    if rule == "mixed":
        return (pair["a"] / r) ** pair["m"], pair["m"]
    return (src[4] / r) ** src[1], src[1]


def evaluate(
    species, pos, cell, periodic, cutoff, params=None, stress=False, density="mixed"
):
    """Return (energy, forces, stress_voigt) as Decimals.

    ``density`` selects the density term. ``"mixed"`` is Eq. (1c) of the
    paper, ``(a_ij / r)^m_ij`` with the mixing rules of Eq. (2). ``"source"``
    uses the neighbour's own ``(a_j / r)^m_j``, the form the pinned eOn QSC
    implements; it exists only so scripts can attribute eOn's deviations.

    ``pos`` and ``cell`` hold Decimals (or exact floats via ``D(float)``);
    ``cell`` rows are the lattice vectors. ``stress_voigt`` is the Cauchy
    stress (1 / V) dU/d(strain) in xx, yy, zz, yz, xz, xy order, or None.
    """
    with _digits():
        return _evaluate(species, pos, cell, periodic, cutoff, params, stress, density)


def _evaluate(species, pos, cell, periodic, cutoff, params, stress, density):
    params = _dec_params(QSC_TABLE if params is None else params)
    cutoff = D(cutoff)
    n_atoms = len(species)
    pairs = {}
    for za in set(species):
        for zb in set(species):
            pairs[(za, zb)] = _pair(params[za], params[zb])

    neigh = [[] for _ in range(n_atoms)]
    rho = [D(0)] * n_atoms
    pair_energy = D(0)
    for i in range(n_atoms):
        for j in range(n_atoms):
            for d in _images(pos[i], pos[j], cell, periodic, cutoff):
                r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                if r2 == 0:
                    continue
                r = r2.sqrt()
                p = pairs[(species[i], species[j])]
                phi, m_i = _density_term(density, p, params[species[j]], r)
                phi_back, m_j = _density_term(density, p, params[species[i]], r)
                v = p["eps"] * (p["a"] / r) ** p["n"]
                rho[i] += phi
                pair_energy += v / 2
                neigh[i].append((j, d, r2, phi, m_i, phi_back, m_j, v, p["n"]))

    embed = [params[species[i]][3] * params[species[i]][2] for i in range(n_atoms)]
    energy = pair_energy - sum(embed[i] * rho[i].sqrt() for i in range(n_atoms))
    # An atom without neighbours has rho = 0 and contributes nothing.
    weight = [
        embed[i] / rho[i].sqrt() if rho[i] > 0 else D(0) for i in range(n_atoms)
    ]

    forces = [[D(0)] * 3 for _ in range(n_atoms)]
    sigma = [[D(0)] * 3 for _ in range(3)]
    for i in range(n_atoms):
        for j, d, r2, phi, m_i, phi_back, m_j, v, n_ij in neigh[i]:
            g = (n_ij * v - (weight[i] * m_i * phi + weight[j] * m_j * phi_back) / 2) / r2
            for k in range(3):
                forces[i][k] += g * d[k]
            if stress:
                for a in range(3):
                    for b in range(3):
                        sigma[a][b] -= g * d[a] * d[b] / 2
    stress_voigt = None
    if stress:
        volume = abs(_det3(cell))
        stress_voigt = [sigma[a][b] / volume for a, b in VOIGT]
    return energy, forces, stress_voigt


def energy_only(species, pos, cell, periodic, cutoff, params=None, density="mixed"):
    with _digits():
        return evaluate(species, pos, cell, periodic, cutoff, params, False, density)[0]


def _strained(pos, cell, a, b, step):
    e = [[D(0)] * 3 for _ in range(3)]
    if a == b:
        e[a][a] = step
    else:
        e[a][b] = e[b][a] = step / 2

    def apply(vec):
        return [vec[i] + sum(vec[k] * e[k][i] for k in range(3)) for i in range(3)]

    return [apply(p) for p in pos], [apply(row) for row in cell]


def selfcheck(verbose=True):
    """Compare the analytic gradient and stress with 50-digit differences."""
    with _digits():
        return all(_selfcheck(verbose, rule) for rule in ("mixed", "source"))


def _selfcheck(verbose, rule):
    rng = random.Random(7)
    species = [29, 47, 28, 79, 46, 78, 45, 77]
    cell = [[D(7.3), D(0), D(0)], [D(0.5), D(7.1), D(0)], [D(-0.4), D(0.3), D(7.6)]]
    pos = []
    corners = list(itertools.product((0, 1), repeat=3))
    for idx in range(len(species)):
        frac = [D((corners[idx][r] + 0.15 * rng.random()) / 2) for r in range(3)]
        pos.append(
            [sum(frac[r] * cell[r][k] for r in range(3)) for k in range(3)]
        )
    cutoff = 3.4
    energy, forces, stress = evaluate(
        species, pos, cell, True, cutoff, stress=True, density=rule
    )
    worst_f = D(0)
    h = D("1e-20")
    for i in (0, 3, 7):
        for k in range(3):
            up = [list(p) for p in pos]
            dn = [list(p) for p in pos]
            up[i][k] += h
            dn[i][k] -= h
            fd = -(
                energy_only(species, up, cell, True, cutoff, density=rule)
                - energy_only(species, dn, cell, True, cutoff, density=rule)
            ) / (2 * h)
            worst_f = max(worst_f, abs(fd - forces[i][k]))
    worst_s = D(0)
    volume = abs(_det3(cell))
    for idx, (a, b) in enumerate(VOIGT):
        up_p, up_c = _strained(pos, cell, a, b, h)
        dn_p, dn_c = _strained(pos, cell, a, b, -h)
        fd = (
            energy_only(species, up_p, up_c, True, cutoff, density=rule)
            - energy_only(species, dn_p, dn_c, True, cutoff, density=rule)
        ) / (2 * h * volume)
        worst_s = max(worst_s, abs(fd - stress[idx]))
    net = max(abs(sum(forces[i][k] for i in range(len(species)))) for k in range(3))
    if verbose:
        print(f"selfcheck[{rule}]: energy {energy}")
        print(f"selfcheck[{rule}]: max |F - FD|      = {float(worst_f):.3e}")
        print(f"selfcheck[{rule}]: max |sigma - FD|  = {float(worst_s):.3e}")
        print(f"selfcheck[{rule}]: |sum F|           = {float(net):.3e}")
    ok = worst_f < D("1e-28") and worst_s < D("1e-28") and net < D("1e-28")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selfcheck", action="store_true", help="finite-difference self test")
    args = parser.parse_args(argv)
    if args.selfcheck:
        with _digits():
            return 0 if selfcheck() else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
