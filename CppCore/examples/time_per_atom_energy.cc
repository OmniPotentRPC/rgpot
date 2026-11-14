// MIT License
// Copyright 2023--present rgpot developers
//
// Cost of per-atom energies: wall time per call of forceImpl against
// forceEnergies on the same geometry, the two interleaved block by block so
// drift in clock, cache or neighbour state hits both alike. Not a meson
// test.
//
//   time_per_atom_energy <system> [blocks] [calls per block]
//
// Systems:
//   cu-bulk  periodic Cu fcc, 13 x 13 x 13 cubic cells (8788 atoms,
//            a = 3.61 A), the Fortran CuH2 EAM kernel, Cu only.
//   ar-bulk  periodic Ar fcc, 5 x 5 x 5 cubic cells (500 atoms, a = 5.26 A),
//            LJPot with u0 = 0.0104 eV, psi = 3.40 A, cutoff = 8.5 A.
//
// Each block times `calls` back-to-back calls of one variant; block pairs
// alternate which variant goes first. Prints one line per variant with the
// median and minimum of the per-block mean, then the median ratio of the
// paired blocks.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench_systems.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace {

using clock_type = std::chrono::steady_clock;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

template <class Pot>
int run(const char *name, const Pot &pot, const rgpot_bench::System &s, int blocks,
        int calls) {
  const std::size_t n = s.types.size();
  std::vector<double> F(3 * n);
  std::vector<double> E(n);
  const rgpot::ForceInput in{n, s.pos.data(), s.types.data(), s.box.data()};
  double plainEnergy = 0.0;
  double sum = 0.0;
  const auto block = [&](bool perAtom) {
    const auto t0 = clock_type::now();
    for (int c = 0; c < calls; ++c) {
      rgpot::ForceOut out{F.data(), 0.0, 0.0, {}, 0};
      if (perAtom) {
        pot.forceEnergies(in, &out, E.data());
        sum = 0.0;
        for (double e : E) {
          sum += e;
        }
        if (out.energy != plainEnergy) {
          std::fprintf(stderr, "%s: per-atom call changed the energy\n", name);
          std::exit(1);
        }
      } else {
        pot.forceImpl(in, &out);
        plainEnergy = out.energy;
      }
    }
    return std::chrono::duration<double>(clock_type::now() - t0).count() / calls;
  };
  block(false);
  block(true);
  std::vector<double> plain;
  std::vector<double> perAtom;
  std::vector<double> ratio;
  for (int b = 0; b < blocks; ++b) {
    double tp = 0.0;
    double te = 0.0;
    if (b % 2 == 0) {
      tp = block(false);
      te = block(true);
    } else {
      te = block(true);
      tp = block(false);
    }
    plain.push_back(tp);
    perAtom.push_back(te);
    ratio.push_back(te / tp);
  }
  const char *threads = std::getenv("OMP_NUM_THREADS");
  std::printf("%s natoms %zu threads %s blocks %d calls %d energy %.10f "
              "sum(energies) %.10f\n",
              name, n, threads ? threads : "default", blocks, calls, plainEnergy, sum);
  std::printf("%s forceImpl     median %.6f ms  min %.6f ms\n", name,
              1e3 * median(plain), 1e3 * *std::min_element(plain.begin(), plain.end()));
  std::printf("%s forceEnergies median %.6f ms  min %.6f ms\n", name,
              1e3 * median(perAtom),
              1e3 * *std::min_element(perAtom.begin(), perAtom.end()));
  std::printf("%s ratio forceEnergies/forceImpl median %.4f  range [%.4f, %.4f]\n", name,
              median(ratio), *std::min_element(ratio.begin(), ratio.end()),
              *std::max_element(ratio.begin(), ratio.end()));
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s cu-bulk|ar-bulk [blocks] [calls]\n", argv[0]);
    return 2;
  }
  const std::string system = argv[1];
  const int blocks = argc > 2 ? std::atoi(argv[2]) : 21;
  if (system == "ar-bulk") {
    const int calls = argc > 3 ? std::atoi(argv[3]) : 200;
    const rgpot::LJPot pot(rgpot::LJConfig{.u0 = 0.0104, .cutoff = 8.5, .psi = 3.40});
    return run("ar-bulk", pot, rgpot_bench::bulk(5, 5.26, 18), blocks, calls);
  }
#ifdef RGPOT_HAS_FORTRAN_POTS
  if (system == "cu-bulk") {
    const int calls = argc > 3 ? std::atoi(argv[3]) : 5;
    const rgpot::fortranpots::CuH2Pot pot;
    return run("cu-bulk", pot, rgpot_bench::bulk(13, 3.61, 29), blocks, calls);
  }
#endif
  std::fprintf(stderr, "unknown system %s\n", system.c_str());
  return 2;
}
