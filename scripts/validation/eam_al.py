# /// script
# requires-python = ">=3.10"
# dependencies = ["sympy>=1.12"]
# ///
# MIT License
# Copyright 2023--present rgpot developers
"""Symbolic checks of the EAM aluminium kernel (rgpot_eam_al.f90).

1. The per-entry density and its slope, built from two shared
   exponentials, equal the shifted closed form and its derivative.
2. The pair energy and slope built from the two pair exponentials and the
   stored density equal the closed form phi(r) and phi'(r).
3. The Horner embedding F(rho) and F'(rho) equal g rho + sum_k c_k rho^k
   and its derivative, for symbolic coefficients.
4. For three atoms with generic phi, rho and F, the gather force
   f_i = sum_j [phi'(r_ij) + (F'(rho_i) + F'(rho_j)) rho'(r_ij)] rhat_ij,
   rhat_ij pointing from i to j, equals -dE/dx_i of
   E = sum_{i<j} phi(r_ij) + sum_i F(rho_i).

Run: uv run scripts/validation/eam_al.py
"""

import sys

import sympy as sp

FAILED = []


def check(name, expr):
    ok = sp.simplify(expr) == 0
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILED.append(name)


r, rc = sp.symbols("r r_c", positive=True)
A, a, B, b = sp.symbols("A a B b", real=True)
s_, da, db, w = sp.symbols("s beta_a beta_b w", real=True)
g = sp.symbols("g", real=True)
eta = 6  # density_power in the shipped parameters

# Closed forms (module header, eOn's Aluminum potential).
dens_shape = lambda x: x**eta * (sp.exp(-da * x) + w * sp.exp(-db * x))
pair_shape = lambda x: A * sp.exp(-a * x) + B * sp.exp(-b * x)
rho_cf = s_ * (dens_shape(r) - dens_shape(rc))
phi_cf = pair_shape(r) - pair_shape(rc) - 2 * g * rho_cf

# Kernel forms (atom_density, atom_contribution).
ea = sp.exp(-da * r)
eb = w * sp.exp(-db * r)
rpm1 = r ** (eta - 1)
rp = rpm1 * r
dens_k = s_ * (rp * (ea + eb) - dens_shape(rc))
ddens_k = s_ * (eta * rpm1 * (ea + eb) - rp * (da * ea + db * eb))
pa = A * sp.exp(-a * r)
pb = B * sp.exp(-b * r)
phi_k = pa + pb - pair_shape(rc) - 2 * g * dens_k
dphi_k = -(a * pa + b * pb) - 2 * g * ddens_k

check("density from shared exponentials", dens_k - rho_cf)
check("density slope", ddens_k - sp.diff(rho_cf, r))
check("pair energy", phi_k - phi_cf)
check("pair slope", dphi_k - sp.diff(phi_cf, r))

# Horner embedding (embed_horner), eighth order.
rho = sp.symbols("rho", real=True)
c = sp.symbols("c1:9", real=True)
F_cf = g * rho + sum(ck * rho ** (k + 1) for k, ck in enumerate(c))
p, dp = c[7], sp.Integer(0)
for k in range(6, -1, -1):
    dp = dp * rho + p
    p = p * rho + c[k]
check("Horner F", rho * (g + p) - F_cf)
check("Horner F'", (g + p + rho * dp) - sp.diff(F_cf, rho))

# Gather force for three atoms, generic functions.
phi = sp.Function("phi")
rhof = sp.Function("rho")
Ff = sp.Function("F")
X = [sp.symbols(f"x{i} y{i} z{i}", real=True) for i in range(3)]


def dist(i, j):
    return sp.sqrt(sum((X[j][k] - X[i][k]) ** 2 for k in range(3)))


E = sum(phi(dist(i, j)) for i in range(3) for j in range(i + 1, 3))
dens_at = [sum(rhof(dist(i, j)) for j in range(3) if j != i) for i in range(3)]
E += sum(Ff(d) for d in dens_at)
Fp = [sp.diff(Ff(rho), rho).subs(rho, d) for d in dens_at]
t = sp.symbols("t")
for k in range(3):
    gather = 0
    for j in range(1, 3):
        rij = dist(0, j)
        dphi = sp.diff(phi(t), t).subs(t, rij)
        drho = sp.diff(rhof(t), t).subs(t, rij)
        gather += (dphi + (Fp[0] + Fp[j]) * drho) * (X[j][k] - X[0][k]) / rij
    check(f"gather force on atom 0, component {k}", gather + sp.diff(E, X[0][k]))

if FAILED:
    print(f"{len(FAILED)} identities failed", file=sys.stderr)
    sys.exit(1)
print("all identities hold")
