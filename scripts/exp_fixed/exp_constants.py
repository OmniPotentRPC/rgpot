"""Exact checks of the exp_fixed constants with sympy rationals."""
import sympy as sp

def rat(h):
    return sp.Rational(*float.fromhex(h).as_integer_ratio())

ln2 = sp.log(2)
hi = rat("0x1.62e42feep-1")
lo = rat("0x1.a39ef35793c76p-33")
inv = rat("0x1.71547652b82fep0")
shift = rat("0x1.8p52")
dmax = (2 - sp.Rational(1, 2**52)) * sp.Integer(2) ** 1023
dmin = sp.Integer(2) ** -1074

# kd * hi exact for |kd| <= 1076: hi has at most 53 - 11 significant bits.
num, den = hi.as_numer_denom()
sig = len(bin(int(num))[2:].rstrip("0"))
print("hi significant bits", sig, "-> kd*hi exact:", sig + 11 <= 53)
print("|ln2 - hi - lo| =", sp.N(abs(ln2 - hi - lo), 5), "(2^%.1f)" % sp.N(sp.log(abs(ln2 - hi - lo), 2)))
print("|1/ln2 - inv| / (1/ln2) =", sp.N(abs(1 / ln2 - inv) * ln2, 5))
# Clamp limits.
print("exp(710) > DBL_MAX:", sp.exp(710) > dmax)
print("exp(-746) < denorm_min/2:", sp.exp(-746) < dmin / 2)
print("exp(-745.1) rounds to denorm_min:", dmin / 2 < sp.exp(sp.Rational("-745.1")) < 3 * dmin / 2)
# Reduced argument range: |x/ln2 - kd| <= 1/2 + rounding of x*inv.
xmax = 746
slack = xmax * abs(1 / ln2 - inv) + sp.Rational(xmax, 1) / ln2 * sp.Rational(1, 2**53)
rmax = ln2 * (sp.Rational(1, 2) + slack)
print("max |r| =", sp.N(rmax, 12), "<= 0.3466:", rmax <= sp.Rational("0.3466"))
# Exponent-field construction: m + 1023 in [485, 1535] fits 11 bits.
for m in (-538, 512):
    v = int(m + shift + 1023)
    print("m", m, "field", (v << 52) >> 52 & 0x7FF, "== m+1023:", ((v << 52) >> 52 & 0x7FF) == m + 1023)
# Coefficients against 1/n!.
cs = ["0x1.000000000000ap-1", "0x1.55555555554fep-3", "0x1.55555555506bcp-5",
      "0x1.1111111126b36p-7", "0x1.6c16c18477083p-10", "0x1.a01a012fee99cp-13",
      "0x1.a019995efb00fp-16", "0x1.71df18ef76144p-19", "0x1.28adfccb26cd8p-22",
      "0x1.ad92e899366cbp-26"]
for n, h in enumerate(cs, start=2):
    print(f"c{n} / (1/{n}!) - 1 = {sp.N(rat(h) * sp.factorial(n) - 1, 3)}")
