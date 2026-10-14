// MIT License
// Copyright 2023--present rgpot developers
//
// exact_fma and mul_add against std::fma, including subnormals, signed
// zero, overflow and ties.

#include "rgpot/simd/exact_fma.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>

namespace {

std::uint64_t bits_of(double x) {
  std::uint64_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  return u;
}

int g_fail = 0;
int g_checked = 0;

void check_one(double a, double b, double c, const char *where) {
  const double ref = std::fma(a, b, c);
  const double got = rgpot::simd::exact_fma(a, b, c);
  const double via = rgpot::simd::mul_add(a, b, c);
  const double quick = rgpot::simd::fma_quick(a, b, c);
#if defined(__GNUC__)
  // Lane 1 pairs with an ordinary case so a fallback on one lane is tested
  // both on its own and next to a lane that stays on the fast path.
  const rgpot::simd::f64x2 pa = {a, 1.5};
  const rgpot::simd::f64x2 pb = {b, -3.25};
  const rgpot::simd::f64x2 pc = {c, 0.125};
  const rgpot::simd::f64x2 px = rgpot::simd::exact_fma_x2(pa, pb, pc);
  const rgpot::simd::f64x2 qx = rgpot::simd::exact_fma_x2(
      rgpot::simd::f64x2{a, a}, rgpot::simd::f64x2{b, b}, rgpot::simd::f64x2{c, c});
  const bool pair_ok = bits_of(px[0]) == bits_of(ref) &&
                       bits_of(px[1]) == bits_of(std::fma(1.5, -3.25, 0.125)) &&
                       bits_of(qx[0]) == bits_of(ref) && bits_of(qx[1]) == bits_of(ref);
#else
  const bool pair_ok = true;
#endif
  ++g_checked;
  if (pair_ok && bits_of(got) == bits_of(ref) && bits_of(via) == bits_of(ref) &&
      bits_of(quick) == bits_of(ref)) {
    return;
  }
  if (g_fail < 8) {
    std::cerr << where << " a=" << bits_of(a) << " b=" << bits_of(b)
              << " c=" << bits_of(c) << " ref=" << bits_of(ref)
              << " exact=" << bits_of(got) << " mul_add=" << bits_of(via)
              << " quick=" << bits_of(quick) << '\n';
  }
  ++g_fail;
}

} // namespace

int main() {
  const double specials[] = {
      0.0,
      -0.0,
      1.0,
      -1.0,
      2.0,
      -2.0,
      0.5,
      -0.5,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),
      0x1p-1022,
      0x1p-1074,
      0x1.fffffffffffffp1023,
      0x1p-500,
      0x1p500,
      3.141592653589793,
      -2.718281828459045,
      0x1.0000000000001p0,
      0x1.fffffffffffffp0,
  };
  const int ns = static_cast<int>(sizeof(specials) / sizeof(specials[0]));
  for (int i = 0; i < ns; ++i) {
    for (int j = 0; j < ns; ++j) {
      for (int k = 0; k < ns; ++k) {
        check_one(specials[i], specials[j], specials[k], "edge");
      }
    }
  }

  std::mt19937_64 rng(1);
  std::uniform_int_distribution<int> expd(-180, 180);
  std::uniform_int_distribution<std::uint64_t> mant(0, (1ULL << 52) - 1);
  std::uniform_int_distribution<int> sign(0, 1);
  auto rnd = [&]() {
    double x = std::ldexp(static_cast<double>(mant(rng) | (1ULL << 52)),
                           expd(rng) - 52);
    if (sign(rng) != 0) {
      x = -x;
    }
    return x;
  };
  for (int i = 0; i < 20000; ++i) {
    const double a = rnd();
    const double b = rnd();
    const double c = rnd();
    check_one(a, b, c, "rand");
    if (std::isfinite(a * b)) {
      check_one(a, b, -a * b, "cancel");
    }
    const double p = a * b;
    if (std::isfinite(p) && p != 0.0) {
      const double ulp = std::scalbn(1.0, std::ilogb(std::fabs(p)) - 52);
      check_one(a, b, 0.5 * ulp, "tie+");
      check_one(a, b, -0.5 * ulp, "tie-");
    }
  }
  // Sparse and dense mantissas over the whole exponent range, with the
  // addend placed on half-ulp ties, just past them, and on near cancellation.
  std::uniform_int_distribution<int> wide(-1100, 1100);
  std::uniform_int_distribution<int> half(-560, 560);
  auto shaped = [&](int e) {
    std::uint64_t m = mant(rng);
    switch (rng() % 4) {
    case 0:
      m &= 0xfff0000000000ULL;
      break;
    case 1:
      m |= 0xffffffULL;
      break;
    default:
      break;
    }
    double x = std::ldexp(static_cast<double>(m | (1ULL << 52)), e - 52);
    return sign(rng) != 0 ? -x : x;
  };
  for (int i = 0; i < 400000; ++i) {
    const double a = shaped(half(rng));
    const double b = shaped(half(rng));
    double c = shaped(wide(rng));
    const double p = a * b;
    switch (i % 5) {
    case 1:
      if (std::isfinite(p) && p != 0.0) {
        const double ulp = std::scalbn(1.0, std::ilogb(p) - 52);
        c = (sign(rng) != 0 ? 0.5 : -0.5) * ulp * (sign(rng) != 0 ? 1.0 : 0.5);
      }
      break;
    case 2:
      c = -p;
      break;
    case 3:
      c = -p + std::ldexp(std::fabs(p), -53 - static_cast<int>(rng() % 60)) *
                   (sign(rng) != 0 ? 1.0 : -1.0);
      break;
    case 4:
      if (std::isfinite(p) && p != 0.0) {
        const double ulp = std::scalbn(1.0, std::ilogb(p) - 52);
        c = std::nextafter(-p, 0.0) + (sign(rng) != 0 ? 0.5 : -0.5) * ulp;
      }
      break;
    default:
      break;
    }
    check_one(a, b, c, "shaped");
  }
  // Edges of the fast-path domain, where the scalar fallback takes over.
  const double band[] = {0x1p-480, 0x1p-479, 0x1.fffffffffffffp-480,
                         0x1p479,  0x1p480,  0x1.fffffffffffffp479,
                         0x1p-900, 0x1p-899, 0x1p899,
                         0x1p900,  0x1p-537, 0x1p-538};
  for (double x : band) {
    for (double y : band) {
      for (double z : band) {
        check_one(x, y, z, "band");
        check_one(-x, y, z, "band");
        check_one(x, y, -z, "band");
      }
    }
  }
  for (int e = -60; e <= 40; e += 5) {
    const double a = std::ldexp(1.0, e);
    const double b = std::ldexp(0x1.0000000000001p0, -e - 30);
    check_one(a, b, -a * b, "under");
  }

  const double tie_a = 0x1.0000000000001p0;
  const double tie_c = 0x1.921fb54442d18p1;
  check_one(tie_a, tie_a, tie_c, "classic-tie");
  const double trunc = rgpot::simd::fma_trunc(tie_a, tie_a, tie_c);
  const double ref = std::fma(tie_a, tie_a, tie_c);
  if (bits_of(trunc) == bits_of(ref)) {
    std::cerr << "classic-tie trunc matched std::fma\n";
    ++g_fail;
  }

  if (g_fail != 0) {
    std::cerr << "fail " << g_fail << " of " << g_checked << '\n';
    return 1;
  }
  std::cout << "ok " << g_checked << '\n';
  return 0;
}
