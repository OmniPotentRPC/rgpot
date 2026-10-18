// MIT License
// Copyright 2023--present rgpot developers
//
// rgpot::simd::exp_fixed: the vector clones against the scalar body bit for
// bit, a digest of 2^20 results that every instruction set and C library
// must reproduce, the special values, and the error against long double.

#include "rgpot/simd/exp.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace {

std::uint64_t bits_of(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

int g_fail = 0;

void expect_bits(double x, double got, double want, const char *what) {
  if (bits_of(got) == bits_of(want)) {
    return;
  }
  if (g_fail < 8) {
    std::cerr << what << " x=" << x << " got=" << bits_of(got)
              << " want=" << bits_of(want) << '\n';
  }
  ++g_fail;
}

/// A double in [lo, hi) from the top 53 bits of one generator output. The
/// standard fixes mt19937_64's sequence but not what uniform_real_distribution
/// or uniform_int_distribution make of it, so the digest's inputs are drawn
/// here, the same on every C++ library.
double uniform(std::mt19937_64 &rng, double lo, double hi) {
  return lo + static_cast<double>(rng() >> 11) * 0x1p-53 * (hi - lo);
}

/// An integer in [lo, hi] from one generator output.
int uniform_int(std::mt19937_64 &rng, int lo, int hi) {
  const std::uint64_t span = static_cast<std::uint64_t>(hi - lo) + 1;
  return lo + static_cast<int>(rng() % span);
}

/// Inputs spread over the whole domain, the reduction boundaries
/// (k ln2 +- ln2/2), and the overflow and underflow edges.
std::vector<double> inputs(std::size_t n) {
  std::mt19937_64 rng(7);
  std::vector<double> x;
  x.reserve(n);
  while (x.size() < n) {
    switch (x.size() % 4) {
    case 0:
      x.push_back(uniform(rng, -750.0, 712.0));
      break;
    case 1:
      x.push_back(uniform(rng, -20.0, 5.0));
      break;
    case 2: {
      const double mid =
          (uniform_int(rng, -1075, 1024) + 0.5) * 0x1.62e42fefa39efp-1;
      x.push_back(std::nextafter(mid, (rng() & 1) ? 1e300 : -1e300));
      break;
    }
    default: {
      std::uint64_t u = rng();
      double v = 0;
      std::memcpy(&v, &u, sizeof(v));
      x.push_back(v);
      break;
    }
    }
  }
  return x;
}

} // namespace

int main() {
  using rgpot::simd::exp_fixed;
  using rgpot::simd::exp_fixed_n;

  // Vector clones against the scalar body, for every tail length.
  const std::vector<double> x = inputs(std::size_t{1} << 20);
  std::vector<double> y(x.size());
  exp_fixed_n(x.data(), y.data(), x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    expect_bits(x[i], y[i], exp_fixed(x[i]), "vector");
  }
  for (std::size_t len = 1; len <= 67; ++len) {
    std::vector<double> part(len);
    exp_fixed_n(x.data() + 3, part.data(), len);
    for (std::size_t i = 0; i < len; ++i) {
      expect_bits(x[3 + i], part[i], y[3 + i], "tail");
    }
  }

  // FNV-1a over the result bits, NaNs folded to one pattern. The constant
  // is the same on x86-64 (any clone) and AArch64; a change in it means the
  // function changed.
  std::uint64_t h = 1469598103934665603ULL;
  for (double v : y) {
    const std::uint64_t u = std::isnan(v) ? 0x7ff8000000000000ULL : bits_of(v);
    for (int b = 0; b < 8; ++b) {
      h ^= (u >> (8 * b)) & 0xffu;
      h *= 1099511628211ULL;
    }
  }
  const std::uint64_t kDigest = 0x13afb36416db71bdULL;
  if (h != kDigest) {
    std::cerr << "digest " << std::hex << h << " expected " << kDigest
              << std::dec << '\n';
    ++g_fail;
  }

  // Special values.
  const double inf = std::numeric_limits<double>::infinity();
  expect_bits(0.0, exp_fixed(0.0), 1.0, "exp(0)");
  expect_bits(-0.0, exp_fixed(-0.0), 1.0, "exp(-0)");
  expect_bits(inf, exp_fixed(inf), inf, "exp(inf)");
  expect_bits(-inf, exp_fixed(-inf), 0.0, "exp(-inf)");
  expect_bits(710.0, exp_fixed(710.0), inf, "overflow");
  expect_bits(-746.0, exp_fixed(-746.0), 0.0, "underflow");
  if (!std::isnan(exp_fixed(std::numeric_limits<double>::quiet_NaN()))) {
    std::cerr << "exp(NaN) is not NaN\n";
    ++g_fail;
  }
  if (!std::isfinite(exp_fixed(709.78)) || exp_fixed(709.79) != inf) {
    std::cerr << "overflow threshold\n";
    ++g_fail;
  }
  if (exp_fixed(-745.1) != std::numeric_limits<double>::denorm_min()) {
    std::cerr << "smallest subnormal result\n";
    ++g_fail;
  }

  // Error in ulps of the result against expl, where long double is wider
  // than double. 1.24 ulp is the a priori bound for normal results (see
  // docs/orgmode/reference/simd_exp.org).
#if LDBL_MANT_DIG > DBL_MANT_DIG
  double worst = 0.0;
  for (double v : x) {
    if (!(v > -708.0 && v < 709.0)) {
      continue;
    }
    const long double ref = std::exp(static_cast<long double>(v));
    const double got = exp_fixed(v);
    const double ulp = std::ldexp(1.0, std::ilogb(static_cast<double>(ref)) - 52);
    const double err =
        static_cast<double>(std::fabs(static_cast<long double>(got) - ref) / ulp);
    worst = std::max(worst, err);
  }
  std::cout << "max error " << worst << " ulp\n";
  if (!(worst <= 1.24)) {
    std::cerr << "error " << worst << " ulp exceeds the bound\n";
    ++g_fail;
  }
#endif

  if (g_fail != 0) {
    std::cerr << "fail " << g_fail << '\n';
    return 1;
  }
  std::cout << "ok digest " << std::hex << h << std::dec << '\n';
  return 0;
}
