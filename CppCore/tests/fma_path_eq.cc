// MIT License
// Copyright 2023--present rgpot developers
//
// Energies and forces of LJ and Morse are the same bits whichever FMA path
// runs. The test hashes a set of evaluations, starts itself again with
// RGPOT_FMA=software, and compares the two digests. It also checks the
// Morse row kernel against a per-pair kernel built from exp_fixed.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/nlist/PairListCache.hpp"
#include "rgpot/simd/exact_fma.hpp"
#include "rgpot/simd/exp.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace {

using Box = std::array<std::array<double, 3>, 3>;

struct Hash {
  std::uint64_t h = 1469598103934665603ULL;
  void add(double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    for (int b = 0; b < 8; ++b) {
      h ^= (u >> (8 * b)) & 0xffu;
      h *= 1099511628211ULL;
    }
  }
};

/// n^3 fcc cells of lattice constant a, every atom moved by up to `jiggle`.
AtomMatrix fcc(int n, double a, double jiggle, std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> d(-jiggle, jiggle);
  const double basis[4][3] = {
      {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  AtomMatrix R(static_cast<std::size_t>(4 * n * n * n), 3);
  std::size_t k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int l = 0; l < n; ++l) {
        for (const auto &b : basis) {
          R(k, 0) = (i + b[0]) * a + d(rng);
          R(k, 1) = (j + b[1]) * a + d(rng);
          R(k, 2) = (l + b[2]) * a + d(rng);
          ++k;
        }
      }
    }
  }
  return R;
}

/// Three evaluations per geometry, each moved a little: a first sighting,
/// the capture, and a cache hit, so every pair-list path runs.
template <typename Pot>
void hashPot(Pot &pot, AtomMatrix R, const Box &box, Hash &h,
             std::mt19937_64 &rng) {
  std::uniform_real_distribution<double> d(-0.01, 0.01);
  const std::vector<int> types(R.rows(), 1);
  for (int rep = 0; rep < 3; ++rep) {
    auto [e, f, v] = pot(R, types, box);
    (void)v;
    h.add(e);
    for (std::size_t i = 0; i < f.rows(); ++i) {
      for (std::size_t c = 0; c < 3; ++c) {
        h.add(f(i, c));
      }
    }
    for (std::size_t i = 0; i < R.rows(); ++i) {
      for (std::size_t c = 0; c < 3; ++c) {
        R(i, c) += d(rng);
      }
    }
  }
}

std::uint64_t digest() {
  std::mt19937_64 rng(11);
  Hash h;
  const double a = 3.92;
  const int n = 6;
  const Box bulk = {{{n * a, 0, 0}, {0, n * a, 0}, {0, 0, n * a}}};
  const Box open = {{{200.0, 0, 0}, {0, 200.0, 0}, {0, 0, 200.0}}};
  rgpot::MorsePot morse;
  rgpot::LJPot lj(rgpot::LJConfig{.u0 = 0.5, .cutoff = 6.0, .psi = 2.3});
  hashPot(morse, fcc(n, a, 0.15, rng), bulk, h, rng);
  hashPot(lj, fcc(n, a, 0.15, rng), bulk, h, rng);
  AtomMatrix cluster = fcc(2, a, 0.3, rng);
  for (std::size_t i = 0; i < cluster.rows(); ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      cluster(i, c) += 90.0;
    }
  }
  hashPot(morse, cluster, open, h, rng);
  hashPot(lj, cluster, open, h, rng);
  return h.h;
}

/// MorsePot (row kernel) against the same pair term evaluated one pair at
/// a time through a private pair-list pool.
int rowAgainstPairs() {
  std::mt19937_64 rng(5);
  const double a = 3.92;
  const int n = 6;
  const Box box = {{{n * a, 0, 0}, {0, n * a, 0}, {0, 0, n * a}}};
  const AtomMatrix R = fcc(n, a, 0.2, rng);
  rgpot::MorsePot pot;
  const std::vector<int> types(R.rows(), 1);
  auto [e, f, v] = pot(R, types, box);
  (void)v;

  const rgpot::MorseConfig cfg = pot.config();
  const double twoDeA = 2.0 * cfg.De * cfg.a;
  const double depth = cfg.De;
  const double range = cfg.a;
  const double rEq = cfg.re;
  const double shiftU = pot.energyShift();
  rgpot::nlist::PairListCache pool;
  rgpot::nlist::CachedPairList::Options opt;
  opt.cutoff = cfg.cutoff;
  std::vector<double> pos(3 * R.rows());
  for (std::size_t i = 0; i < R.rows(); ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      pos[3 * i + c] = R(i, c);
    }
  }
  const double cell[9] = {n * a, 0, 0, 0, n * a, 0, 0, 0, n * a};
  std::vector<double> F(pos.size(), 0.0);
  const double U = pool.accumulate(
      pos.data(), R.rows(), cell, opt, F.data(),
      [=](int32_t, int32_t, double r2) noexcept {
        const double r = std::sqrt(r2);
        const double d = 1.0 - rgpot::simd::exp_fixed(-range * (r - rEq));
        return rgpot::nlist::PairTerm{depth * d * d - depth - shiftU,
                                      twoDeA * d * (d - 1.0) / r};
      });
  int bad = 0;
  Hash a1;
  Hash a2;
  a1.add(e);
  a2.add(U);
  for (std::size_t i = 0; i < R.rows(); ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      a1.add(f(i, c));
      a2.add(F[3 * i + c]);
    }
  }
  if (a1.h != a2.h) {
    std::cerr << "Morse row kernel differs from the per-pair kernel\n";
    ++bad;
  }
  return bad;
}

} // namespace

int main(int argc, char **argv) {
  const std::uint64_t mine = digest();
  if (argc > 1 && std::string(argv[1]) == "--child") {
    std::printf("%016llx %d\n", static_cast<unsigned long long>(mine),
                rgpot::simd::detail::fma_hw_active() ? 1 : 0);
    return 0;
  }
  int fail = rowAgainstPairs();
  const bool hw = rgpot::simd::detail::fma_hw_active();
#if defined(__unix__) || defined(__APPLE__)
  setenv("RGPOT_FMA", "software", 1);
  const std::string cmd = std::string("\"") + argv[0] + "\" --child";
  FILE *p = popen(cmd.c_str(), "r");
  unsigned long long other = 0;
  int otherHw = -1;
  if (p == nullptr || std::fscanf(p, "%llx %d", &other, &otherHw) != 2) {
    std::cerr << "could not run the software-path child\n";
    return 1;
  }
  if (pclose(p) != 0) {
    std::cerr << "software-path child failed\n";
    return 1;
  }
  if (otherHw != 0) {
    std::cerr << "RGPOT_FMA=software left the hardware path on\n";
    ++fail;
  }
  std::cout << "this process: " << (hw ? "hardware" : "software")
            << " fma, digest " << std::hex << mine << "; software child "
            << other << std::dec << '\n';
  if (other != mine) {
    std::cerr << "digests differ between the FMA paths\n";
    ++fail;
  }
#else
  std::cout << "digest " << std::hex << mine << std::dec << '\n';
#endif
  return fail == 0 ? 0 : 1;
}
