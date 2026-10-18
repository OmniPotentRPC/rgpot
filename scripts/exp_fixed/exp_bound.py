"""First-order rounding-error bound for rgpot::simd::exp_fixed.

Every operation obeys fl(a op b) = (a op b)(1 + d), |d| <= u = 2^-53. The
running-error bound of each step is propagated over a grid of reduced
arguments r, and the worst case in ulps of the result is reported. The
grid spacing is far below the scale on which the bound changes.
"""
import math

u = 2.0 ** -53
C = [float.fromhex(h) for h in [
    "0x1.000000000000ap-1", "0x1.55555555554fep-3", "0x1.55555555506bcp-5",
    "0x1.1111111126b36p-7", "0x1.6c16c18477083p-10", "0x1.a01a012fee99cp-13",
    "0x1.a019995efb00fp-16", "0x1.71df18ef76144p-19", "0x1.28adfccb26cd8p-22",
    "0x1.ad92e899366cbp-26"]]          # c2 .. c11
APPROX = 3.7966288726178557e-18         # sollya supnorm, relative
LN2HI = float.fromhex("0x1.62e42feep-1")
LN2LO = float.fromhex("0x1.a39ef35793c76p-33")
KMAX = 1076
# |kd * (ln2 - LN2HI - LN2LO)|: LN2LO is ln2 - LN2HI rounded, error <= ulp/2.
SPLIT = KMAX * 2.0 ** (math.frexp(LN2LO)[1] - 1 - 53)
RMAX = 0.3466

def bound(r):
    # Reduction: r = fl(r0 - fl(kd*LN2LO)), r0 exact.
    dr = u * abs(r) + KMAX * LN2LO * u + SPLIT
    # Horner for q.
    q, eq = C[-1], 0.0
    for c in reversed(C[:-1]):
        prod = q * r
        eq = eq * abs(r) + u * abs(prod)
        q = prod + c
        eq += u * abs(q)
    t2 = r * r
    et2 = u * t2
    w = t2 * q
    ew = et2 * abs(q) + t2 * eq + u * abs(w)
    s = r + w
    es = ew + u * abs(s)
    p = 1.0 + s
    e = math.exp(r)
    err = es + e * (APPROX + dr) + abs(p) * 0.0
    ulp = 2.0 ** (math.frexp(p)[1] - 53)
    return err / ulp + 0.5, err / ulp

N = 200001
worst = max(bound(-RMAX + 2 * RMAX * i / (N - 1)) for i in range(N))
print(f"bound on normal results: {worst[0]:.4f} ulp "
      f"({worst[1]:.4f} ulp before the last rounding)")
