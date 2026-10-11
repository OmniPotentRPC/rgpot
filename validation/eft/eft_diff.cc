// MIT License
// Copyright 2023--present rgpot developers

// Differential check of the error-free transformations an exact FMA
// emulator uses. Knuth's TwoSum and Dekker's product split.
// Ogita, Rump, Oishi, SIAM J. Sci. Comput. 26, 1955 (2005),
// doi:10.1137/030601818.
// Boldo and Muller, IEEE Trans. Comput. 60, 176 (2011),
// doi:10.1109/tc.2010.139.
//
// perf/highway-dispatch does not yet contain a software FMA emulator.
// This file checks the transformations themselves against std::fma and
// against the exact sum or product in binary128.

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <quadmath.h>
#include <random>

struct SumErr {
  double s;
  double e;
};

SumErr two_sum(double a, double b) {
  const double s = a + b;
  const double ap = s - b;
  const double bp = s - ap;
  const double da = a - ap;
  const double db = b - bp;
  return {s, da + db};
}

void dekker_split(double a, double &hi, double &lo) {
  const double c = 134217729.0 * a; // 2^27 + 1
  hi = c - (c - a);
  lo = a - hi;
}

struct ProdErr {
  double p;
  double e;
};

ProdErr two_prod(double a, double b) {
  const double p = a * b;
  double ahi = 0, alo = 0, bhi = 0, blo = 0;
  dekker_split(a, ahi, alo);
  dekker_split(b, bhi, blo);
  const double e = alo * blo - (((p - ahi * bhi) - alo * bhi) - ahi * blo);
  return {p, e};
}

bool finite_pair(double a, double b) {
  return std::isfinite(a) && std::isfinite(b);
}

bool sum_overflows(double a, double b) {
  const double s = a + b;
  return !std::isfinite(s);
}

int g_fail = 0;

void expect(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "fail: %s\n", what);
    ++g_fail;
  }
}

bool quad_equal(__float128 x, __float128 y) { return x == y; }

void check_two_sum(double a, double b, const char *tag) {
  if (!finite_pair(a, b) || sum_overflows(a, b)) {
    return;
  }
  const SumErr r = two_sum(a, b);
  if (!std::isfinite(r.s) || !std::isfinite(r.e)) {
    std::fprintf(stderr, "fail: TwoSum nonfinite %s\n", tag);
    ++g_fail;
    return;
  }
  const __float128 got =
      static_cast<__float128>(r.s) + static_cast<__float128>(r.e);
  const __float128 exact =
      static_cast<__float128>(a) + static_cast<__float128>(b);
  if (!quad_equal(got, exact)) {
    char buf[128];
    quadmath_snprintf(buf, sizeof(buf), "%.20Qe", got - exact);
    std::fprintf(stderr, "fail: TwoSum %s residual %s\n", tag, buf);
    ++g_fail;
  }
}

bool normal_range(double a) {
  return std::isfinite(a) && std::fpclassify(a) == FP_NORMAL &&
         std::fabs(a) >= 0x1p-400 && std::fabs(a) <= 0x1p400;
}

void check_two_prod(double a, double b, const char *tag, int &sub_mismatch) {
  if (!finite_pair(a, b)) {
    return;
  }
  const ProdErr r = two_prod(a, b);
  const double fused = std::fma(a, b, -r.p);
  const bool sub = std::fpclassify(a) == FP_SUBNORMAL ||
                   std::fpclassify(b) == FP_SUBNORMAL ||
                   std::fpclassify(r.p) == FP_SUBNORMAL;
  if (!std::isfinite(r.p)) {
    return;
  }
  if (sub || !normal_range(a) || !normal_range(b)) {
    if (r.e != fused) {
      ++sub_mismatch;
    }
    return;
  }
  if (r.e != fused) {
    std::fprintf(stderr, "fail: TwoProd %s disagrees with fma\n", tag);
    ++g_fail;
    return;
  }
  const __float128 got =
      static_cast<__float128>(r.p) + static_cast<__float128>(r.e);
  const __float128 exact =
      static_cast<__float128>(a) * static_cast<__float128>(b);
  if (!quad_equal(got, exact)) {
    std::fprintf(stderr, "fail: TwoProd %s residual\n", tag);
    ++g_fail;
  }
}

double bits_to_double(std::uint64_t u) {
  double x = 0;
  std::memcpy(&x, &u, sizeof(x));
  return x;
}

int main() {
  std::fesetround(FE_TONEAREST);

  const double pos0 = 0.0;
  const double neg0 = -0.0;
  const SumErr zz = two_sum(neg0, pos0);
  expect(zz.s == 0.0 && !std::signbit(zz.s), "TwoSum(-0, +0) is +0");
  expect(zz.e == 0.0, "TwoSum(-0, +0) error is zero");
  const SumErr zn = two_sum(pos0, neg0);
  expect(zn.s == 0.0 && !std::signbit(zn.s), "TwoSum(+0, -0) is +0");

  const double special[] = {
      pos0,
      neg0,
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      1.0,
      -1.0,
      std::nextafter(1.0, 2.0),
      std::nextafter(1.0, 0.0),
      0.5,
      1.5,
      2.0,
      3.0,
      std::ldexp(1.0, -54),
      std::ldexp(1.0, -53),
      std::numeric_limits<double>::max() / 4.0,
      -std::numeric_limits<double>::max() / 4.0,
  };
  const int ns = static_cast<int>(sizeof(special) / sizeof(special[0]));
  for (int i = 0; i < ns; ++i) {
    for (int j = 0; j < ns; ++j) {
      check_two_sum(special[i], special[j], "special");
      int ignored = 0;
      check_two_prod(special[i], special[j], "special", ignored);
    }
  }

  // A tie for round-to-nearest-even: 1 + 2^-53 is the midpoint between
  // 1 and the next binary64, and TwoSum must return an exact pair.
  check_two_sum(1.0, std::ldexp(1.0, -53), "tie");
  check_two_sum(1.0, std::ldexp(1.0, -52), "ulp");

  std::mt19937_64 rng(0x6070);
  int sub_mismatch = 0;
  int random_n = 0;
  for (int i = 0; i < 200000; ++i) {
    const double a = bits_to_double(rng());
    const double b = bits_to_double(rng());
    if (!finite_pair(a, b)) {
      continue;
    }
    check_two_sum(a, b, "random");
    check_two_prod(a, b, "random", sub_mismatch);
    ++random_n;
  }

  // Overflow is finite input, infinite sum. The transform is not claimed
  // there. It must not trap.
  const SumErr ov = two_sum(std::numeric_limits<double>::max(),
                            std::numeric_limits<double>::max());
  expect(!std::isfinite(ov.s), "TwoSum overflow is nonfinite");
  const ProdErr op = two_prod(1.0e200, 1.0e200);
  expect(!std::isfinite(op.p), "TwoProd overflow is nonfinite");

  const double fma_zero = std::fma(-0.0, 1.0, 0.0);
  expect(fma_zero == 0.0 && !std::signbit(fma_zero), "fma(-0, 1, +0) is +0");

  if (g_fail != 0) {
    std::fprintf(stderr, "eft_diff failures %d\n", g_fail);
    return 1;
  }
  std::printf("eft_diff ok random %d subnormal_or_extreme_prod_mismatches %d\n",
              random_n, sub_mismatch);
  return 0;
}
