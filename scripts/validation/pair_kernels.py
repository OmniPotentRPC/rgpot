# /// script
# requires-python = ">=3.10"
# dependencies = ["sympy>=1.12"]
# ///
# MIT License
# Copyright 2023--present rgpot developers
"""Symbolic checks of the classical pair kernels.

Each kernel returns a pair energy and ``fscale = -V'(r) / r``, and the force
loop adds ``fscale * (r_i - r_j)`` to atom i. For every pair potential this
script checks, with sympy and no floating point:

1. the energy expression the C++ kernel evaluates equals the closed form;
2. the kernel's ``fscale`` equals ``-V'(r) / r``;
3. in Cartesian coordinates, ``fscale * (x_i - x_j)`` equals ``-dE/dx_i``;
4. the quintic switch of PairSwitch.hpp is the unique C^2 quintic, and the
   switched pair term ``V S`` is C^2 at the switch start and vanishes to
   second order at the cutoff, with the kernel's switched ``fscale``;
5. the ZBL switching coefficients (LAMMPS form) zero the energy, force and
   curvature at the outer cutoff and join C^2 at the inner one.

Run: ``uv run scripts/validation/pair_kernels.py``. Exits non-zero on the
first failed identity.
"""

import sys

import sympy as sp

r, psi, u0 = sp.symbols("r psi u0", positive=True)
De, a, re = sp.symbols("D_e a r_e", positive=True)
ron, rc = sp.symbols("r_on r_c", positive=True)
t = sp.symbols("t", real=True)

FAILED = []


def check(name, expr):
    """Require ``expr`` to simplify to zero."""
    ok = sp.simplify(sp.expand(expr)) == 0
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILED.append(name)


# --- Lennard-Jones (LJPot / LJClusterPot) ---------------------------------
V_lj = 4 * u0 * ((psi / r) ** 12 - (psi / r) ** 6)
r2 = r**2
inv_r2 = 1 / r2
sr2 = psi**2 * inv_r2
a6 = sr2**3  # (psi / r)^6 without pow(), as the kernel forms it
b = 4 * u0 * a6
E_lj_code = b * (a6 - 1)
fs_lj_code = 6 * b * inv_r2 * (2 * a6 - 1)
check("LJ energy expression", E_lj_code - V_lj)
check("LJ fscale = -V'/r", fs_lj_code - (-sp.diff(V_lj, r) / r))

# --- Morse (MorsePot) -----------------------------------------------------
d = 1 - sp.exp(-a * (r - re))
V_morse = De * (1 - sp.exp(-a * (r - re))) ** 2 - De
E_morse_code = De * d * d - De
fs_morse_code = 2 * De * a * d * (d - 1) / r
check("Morse energy expression", E_morse_code - V_morse)
check("Morse fscale = -V'/r", fs_morse_code - (-sp.diff(V_morse, r) / r))

# --- Cartesian gradient for one pair in 3D --------------------------------
xi = sp.symbols("x_i y_i z_i", real=True)
xj = sp.symbols("x_j y_j z_j", real=True)
dist = sp.sqrt(sum((p - q) ** 2 for p, q in zip(xi, xj)))
for name, V, fs in (
    ("LJ", V_lj, fs_lj_code),
    ("Morse", V_morse, fs_morse_code),
):
    E = V.subs(r, dist)
    fs_x = fs.subs(r, dist)
    for k in range(3):
        check(
            f"{name} F_i[{k}] = -dE/dx_i[{k}]",
            -sp.diff(E, xi[k]) - fs_x * (xi[k] - xj[k]),
        )
        # Newton's third law: the half list subtracts the same vector from j.
        check(
            f"{name} F_j[{k}] = -F_i[{k}]",
            sp.diff(E, xj[k]) + sp.diff(E, xi[k]),
        )

# --- Quintic switch (PairSwitch.hpp) --------------------------------------
c = sp.symbols("c0:6")
S_gen = sum(ck * t**k for k, ck in enumerate(c))
conds = [
    S_gen.subs(t, 0) - 1,
    sp.diff(S_gen, t).subs(t, 0),
    sp.diff(S_gen, t, 2).subs(t, 0),
    S_gen.subs(t, 1),
    sp.diff(S_gen, t).subs(t, 1),
    sp.diff(S_gen, t, 2).subs(t, 1),
]
sol = sp.solve(conds, c, dict=True)
if len(sol) != 1:
    print("FAIL quintic switch: C^2 conditions do not fix a unique quintic")
    FAILED.append("quintic unique")
else:
    coeffs = [sol[0][ck] for ck in c]
    print(f"     quintic coefficients c0..c5 = {coeffs}")
    for k, (got, want) in enumerate(zip(coeffs, [1, 0, 0, -10, 15, -6])):
        check(f"quintic coefficient c{k}", got - want)

S_horner = 1 + t**2 * t * (-10 + t * (15 - 6 * t))  # PairSwitch.hpp
dS_code = -30 * t**2 * (1 - t) ** 2
S_ref = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
check("switch Horner form", S_horner - S_ref)
check("switch derivative form", dS_code - sp.diff(S_ref, t))

# Switched pair term V(r) S(t(r)) for an arbitrary smooth V.
Vf = sp.Function("V")
tr = (r - ron) / (rc - ron)
E_sw = Vf(r) * S_ref.subs(t, tr)
for k in range(3):
    dk = sp.diff(E_sw, r, k)
    check(f"switched term: d^{k}/dr^{k} vanishes at r_c", dk.subs(r, rc).doit())
    check(
        f"switched term: d^{k}/dr^{k} joins V at r_on",
        (dk - sp.diff(Vf(r), r, k)).subs(r, ron).doit(),
    )
# Kernel form: fscale = fs_raw S - V dS/dr / r, with fs_raw = -V'/r.
dSdr = dS_code.subs(t, tr) / (rc - ron)
fs_sw_code = (-sp.diff(Vf(r), r) / r) * S_ref.subs(t, tr) - Vf(r) * dSdr / r
check("switched fscale = -(V S)'/r", fs_sw_code - (-sp.diff(E_sw, r) / r))

# --- ZBL switching (ZBLPot::buildTables, LAMMPS pair_zbl) -----------------
fc, fcp, fcpp, rin = sp.symbols("f_c fp_c fpp_c r_in", real=True)
tc = rc - rin
swa = (-3 * fcp + tc * fcpp) / tc**2
swb = (2 * fcp - tc * fcpp) / tc**3
swc = -fc + (tc / 2) * fcp - (tc**2 / 12) * fcpp
# Second-order Taylor model of the unswitched energy about r_c: the switch
# conditions involve only f, f', f'' at r_c.
fz = fc + fcp * (r - rc) + fcpp * (r - rc) ** 2 / 2
ts = r - rin
E_zbl = fz + swc + ts**3 * (swa / 3 + swb / 4 * ts)
check("ZBL energy vanishes at cut_global", E_zbl.subs(r, rc))
check("ZBL force vanishes at cut_global", sp.diff(E_zbl, r).subs(r, rc))
check("ZBL curvature vanishes at cut_global", sp.diff(E_zbl, r, 2).subs(r, rc))
dE_code = sp.diff(fz, r) + ts**2 * (swa + swb * ts)  # dzbldr + sw1, sw2 terms
check("ZBL kernel derivative", dE_code - sp.diff(E_zbl, r))
for k in (1, 2):
    check(
        f"ZBL switch term d^{k}/dr^{k} is zero at cut_inner",
        sp.diff(ts**3 * (swa / 3 + swb / 4 * ts), r, k).subs(r, rin),
    )

if FAILED:
    print(f"{len(FAILED)} identities failed", file=sys.stderr)
    sys.exit(1)
print("all identities hold")
