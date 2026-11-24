// MIT License
// Copyright 2023--present rgpot developers
//
// Time of a force-only forceImpl call, in nanoseconds, for the potentials that
// write a stress on request. The calls never ask for stress, so the numbers
// are what a caller that does not want it pays. The program only uses the
// four-field ForceInput and the five-field ForceOut that every release has, so
// the same source builds against an older tree and the numbers compare.
//
// Output, one line per repeat:
//   bench <potential> <n_atoms> <repeat> <calls> <ns_per_call>
//
// Set OMP_NUM_THREADS=1 and pin the process to one core to compare two builds.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"

namespace {

struct Crystal {
  std::vector<double> pos;
  std::array<double, 9> box{};
  std::vector<int> z;
};

Crystal fcc(double lattice, int cells, int atomic_number) {
  const double basis[4][3] = {
      {0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  Crystal c;
  std::uint64_t state = 0x9e3779b97f4a7c15ULL;
  auto jitter = [&]() {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t v = state;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    v ^= v >> 31;
    return (static_cast<double>(v >> 11) * (1.0 / 9007199254740992.0) - 0.5) *
           0.1;
  };
  for (int ix = 0; ix < cells; ++ix) {
    for (int iy = 0; iy < cells; ++iy) {
      for (int iz = 0; iz < cells; ++iz) {
        for (const auto &b : basis) {
          c.pos.push_back((ix + b[0]) * lattice + jitter());
          c.pos.push_back((iy + b[1]) * lattice + jitter());
          c.pos.push_back((iz + b[2]) * lattice + jitter());
          c.z.push_back(atomic_number);
        }
      }
    }
  }
  const double edge = cells * lattice;
  c.box = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  return c;
}

template <class Pot>
void bench(const char *name, const Pot &pot, const Crystal &c) {
  const std::size_t n = c.z.size();
  std::vector<double> forces(3 * n);
  auto call = [&]() {
    std::fill(forces.begin(), forces.end(), 0.0);
    rgpot::ForceOut out{forces.data(), 0.0, 0.0, {}, 0};
    pot.forceImpl({n, c.pos.data(), c.z.data(), c.box.data()}, &out);
    return out.energy;
  };
  double sink = 0.0;
  for (int i = 0; i < 20; ++i) {
    sink += call();
  }
  constexpr double kSeconds = 0.4;
  for (int repeat = 1; repeat <= 9; ++repeat) {
    long calls = 0;
    const auto t0 = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    do {
      for (int i = 0; i < 10; ++i) {
        sink += call();
      }
      calls += 10;
      elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0)
                    .count();
    } while (elapsed < kSeconds);
    std::printf("bench %s %zu %d %ld %.1f\n", name, n, repeat, calls,
                1e9 * elapsed / static_cast<double>(calls));
  }
  if (sink == 0.5) {
    std::printf("unreachable\n");
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::string only = argc > 1 ? argv[1] : "";
  const auto wanted = [&](const char *name) {
    return only.empty() || only == name;
  };
  if (wanted("LJ")) {
    const auto c = fcc(1.6, 6, 18);
    bench("LJ", rgpot::LJPot(rgpot::LJConfig{.cutoff = 3.0}), c);
  }
  if (wanted("Morse")) {
    const auto c = fcc(3.92, 6, 78);
    bench("Morse", rgpot::MorsePot(), c);
  }
  if (wanted("EAM")) {
    const auto c = fcc(4.05, 6, 13);
    bench("EAM", rgpot::EAMPot(), c);
  }
  if (wanted("EMT")) {
    const auto c = fcc(3.61, 6, 29);
    bench("EMT", rgpot::EMTPot(), c);
  }
  if (wanted("SuttonChen")) {
    const auto c = fcc(3.92, 6, 78);
    rgpot::SuttonChenConfig config;
    config.epsilon = 0.01;
    config.a = 3.92;
    config.cutoff = 7.0;
    config.periodic = true;
    bench("SuttonChen", rgpot::SuttonChenPot(config), c);
  }
  return 0;
}
