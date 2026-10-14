// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time of LJPot / LJClusterPot against the kokkos-rs force entry
// on the ar-np and ar-cl systems from scripts/bench_pair_scaling.sh.
// Same modes, call counts, and five repeats. Compiled -O3 for the
// measurement; a debug meson build of this file is not that number.
//
// One row per kernel. `us_per_call_*` is the canonical pair-scaling
// column. parallel_cost_us is threads_used times the median.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "bench_systems.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJKokkos.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

namespace {

using rgpot_bench::System;
using rgpot_bench::nanoparticle;
using clock_type = std::chrono::steady_clock;

constexpr double kWalkBound = 0.3;
constexpr double kU0 = 0.0104;
constexpr double kPsi = 3.40;
constexpr double kCut = 8.5;

struct Row {
  std::string kernel;
  double median = 0;
  double lo = 0;
  double hi = 0;
  double energy = 0;
  std::uint64_t force_calls = 0;
  std::uint64_t pairs = 0;
  std::uint64_t tile_bytes = 0;
  std::uint32_t threads_used = 1;
};

double posSum(const std::vector<double> &pos) {
  double s = 0;
  for (double x : pos)
    s += x;
  return s;
}

long callsFor(std::size_t n) {
  if (n >= 10000)
    return 20;
  if (n >= 1000)
    return 200;
  return 2000;
}

void coldShift(std::vector<double> &pos, const std::vector<double> &origin,
               long call, int rep) {
  const double shift = static_cast<double>((call + 17L * rep) % 64);
  for (std::size_t k = 0; k < pos.size(); k += 3)
    pos[k] = origin[k] + shift;
}

void walkStep(std::vector<double> &pos, const std::vector<double> &origin,
              std::mt19937_64 &rng,
              std::uniform_real_distribution<double> &step) {
  for (std::size_t k = 0; k < pos.size(); ++k) {
    double x = pos[k] + step(rng);
    const double off = x - origin[k];
    if (off > kWalkBound)
      x = origin[k] + 2.0 * kWalkBound - off;
    else if (off < -kWalkBound)
      x = origin[k] - 2.0 * kWalkBound - off;
    pos[k] = x;
  }
}

template <typename Call>
Row timeKernel(const std::string &kernel, const System &s, const std::string &mode,
               long calls, int repeats, std::uint32_t threads_used,
               std::uint64_t pairs, std::uint64_t tile_bytes, Call &&call) {
  std::mt19937_64 rng(1234);
  std::uniform_real_distribution<double> step(-0.02, 0.02);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  double energy = 0;
  const auto origin = s.pos;
  for (int r = 0; r < repeats; ++r) {
    auto pos = origin;
    std::vector<double> force(3 * s.types.size(), 0.0);
    if (mode != "cold") {
      call(pos, force);
      call(pos, force);
    }
    const auto t0 = clock_type::now();
    for (long c = 0; c < calls; ++c) {
      if (mode == "cold")
        coldShift(pos, origin, c, r);
      else if (mode == "walk")
        walkStep(pos, origin, rng, step);
      energy = call(pos, force);
    }
    const auto t1 = clock_type::now();
    samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                      static_cast<double>(calls));
  }
  std::sort(samples.begin(), samples.end());
  Row row;
  row.kernel = kernel;
  row.median = samples[samples.size() / 2];
  row.lo = samples.front();
  row.hi = samples.back();
  row.energy = energy;
  const long warm = mode == "cold" ? 0L : 2L;
  row.force_calls = static_cast<std::uint64_t>(calls + warm) *
                    static_cast<std::uint64_t>(repeats);
  row.pairs = pairs;
  row.tile_bytes = tile_bytes;
  row.threads_used = threads_used;
  return row;
}

struct Forces {
  double energy = 0;
  std::vector<double> f;
};

Forces ljpotOnce(const System &s, bool cluster) {
  std::vector<double> f(3 * s.types.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = s.types.size(),
                       .pos = s.pos.data(),
                       .atmnrs = s.types.data(),
                       .box = s.box.data()};
  rgpot::ForceOut out{.F = f.data(),
                      .energy = 0,
                      .variance = 0,
                      .stress = {},
                      .has_stress = 0};
  if (cluster) {
    rgpot::LJClusterPot pot(
        rgpot::LJClusterConfig{.u0 = kU0, .cutoff = kCut, .psi = kPsi});
    pot.forceImpl(in, &out);
  } else {
    rgpot::LJPot pot(rgpot::LJConfig{.u0 = kU0, .cutoff = kCut, .psi = kPsi});
    pot.forceImpl(in, &out);
  }
  return {out.energy, std::move(f)};
}

Forces krsOnce(rgpot::LJKokkos &pot, const System &s) {
  std::vector<double> f(3 * s.types.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = s.types.size(),
                       .pos = s.pos.data(),
                       .atmnrs = s.types.data(),
                       .box = s.box.data()};
  rgpot::ForceOut out{.F = f.data(),
                      .energy = 0,
                      .variance = 0,
                      .stress = {},
                      .has_stress = 0};
  pot.forceImpl(in, &out);
  return {out.energy, std::move(f)};
}

double maxAbsDiff(const std::vector<double> &a, const std::vector<double> &b) {
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

} // namespace

int main(int argc, char **argv) {
  std::string system = "ar-np";
  std::string mode = "warm";
  std::size_t n = 1000;
  long calls = -1;
  int repeats = 5;
  std::size_t threads = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--system")
      system = next();
    else if (a == "--mode")
      mode = next();
    else if (a == "--n")
      n = std::strtoull(next(), nullptr, 10);
    else if (a == "--calls")
      calls = std::strtol(next(), nullptr, 10);
    else if (a == "--repeats")
      repeats = std::atoi(next());
    else if (a == "--threads")
      threads = std::strtoul(next(), nullptr, 10);
    else {
      std::fprintf(stderr,
                   "usage: time_lj_kokkos --system ar-np|ar-cl --mode "
                   "cold|warm|walk --n N [--calls K] [--repeats R] "
                   "[--threads T]\n");
      return 2;
    }
  }
  if (system != "ar-np" && system != "ar-cl") {
    std::fprintf(stderr, "time_lj_kokkos only times ar-np and ar-cl\n");
    return 2;
  }
  if (calls < 0)
    calls = callsFor(n);
  const bool cluster = system == "ar-cl";
  const System s = nanoparticle(n, 5.26, 9.5, 18);
  std::fprintf(stderr, "pos_sum=%.12f natoms=%zu\n", posSum(s.pos),
               s.types.size());

  const Forces reference = ljpotOnce(s, cluster);
  // One Kokkos runtime per process. The thread count is this invocation's.
  rgpot::LJKokkos kokkos(kU0, kPsi, kCut, !cluster, threads, 0);
  const Forces got = krsOnce(kokkos, s);
  const double eref = std::max(1.0, std::abs(reference.energy));
  const double rel = std::abs(reference.energy - got.energy) / eref;
  const double fmax = maxAbsDiff(reference.f, got.f);
  std::fprintf(stderr, "check rel=%.3e force=%.3e pairs=%llu threads=%zu\n", rel,
               fmax, static_cast<unsigned long long>(kokkos.pairCount()), threads);
  const double rel_lim = threads <= 1 ? 1e-8 : 1e-5;
  const double force_lim = threads <= 1 ? 1e-6 : 1e-4;
  if (rel > rel_lim || fmax > force_lim) {
    std::fprintf(stderr, "Kokkos force entry does not match LJPot/LJClusterPot\n");
    return 3;
  }

  auto invokePot = [&](bool asCluster) {
    return timeKernel(
        "ljpot", s, mode, calls, repeats, 1, 0, 0,
        [&](const std::vector<double> &pos, std::vector<double> &force) {
          rgpot::ForceInput in{.nAtoms = s.types.size(),
                               .pos = pos.data(),
                               .atmnrs = s.types.data(),
                               .box = s.box.data()};
          rgpot::ForceOut out{.F = force.data(),
                              .energy = 0,
                              .variance = 0,
                              .stress = {},
                              .has_stress = 0};
          if (asCluster) {
            static rgpot::LJClusterPot pot(
                rgpot::LJClusterConfig{.u0 = kU0, .cutoff = kCut, .psi = kPsi});
            pot.forceImpl(in, &out);
          } else {
            static rgpot::LJPot pot(
                rgpot::LJConfig{.u0 = kU0, .cutoff = kCut, .psi = kPsi});
            pot.forceImpl(in, &out);
          }
          return out.energy;
        });
  };
  // The static pot above would keep one list across systems. This driver
  // times one system per process, so the static cache is that system.
  Row lj = invokePot(cluster);
  auto invokeKrs = [&](rgpot::LJKokkos &pot, const char *name) {
    const auto before = pot.forceCalls();
    Row row = timeKernel(
        name, s, mode, calls, repeats, pot.threadsUsed(), pot.pairCount(),
        pot.tileBytes(),
        [&](const std::vector<double> &pos, std::vector<double> &force) {
          rgpot::ForceInput in{.nAtoms = s.types.size(),
                               .pos = pos.data(),
                               .atmnrs = s.types.data(),
                               .box = s.box.data()};
          rgpot::ForceOut out{.F = force.data(),
                              .energy = 0,
                              .variance = 0,
                              .stress = {},
                              .has_stress = 0};
          pot.forceImpl(in, &out);
          return out.energy;
        });
    row.force_calls = pot.forceCalls() - before;
    row.pairs = pot.pairCount();
    row.tile_bytes = pot.tileBytes();
    row.threads_used = pot.threadsUsed();
    return row;
  };
  // threadsUsed() before any parallel evaluate may be 0. Time first, then
  // the row records the last call.
  Row krs = invokeKrs(kokkos, "kokkos");

  std::printf(
      "system,mode,natoms,calls,repeats,us_per_call_median,us_per_call_min,"
      "us_per_call_max,energy_last,kernel,force_calls,pairs,tile_bytes,"
      "threads_used,parallel_cost_us\n");
  const Row rows[2] = {lj, krs};
  for (const Row &row : rows) {
    std::printf("%s,%s,%zu,%ld,%d,%.3f,%.3f,%.3f,%.12g,%s,%llu,%llu,%llu,%u,"
                "%.3f\n",
                system.c_str(), mode.c_str(), s.types.size(), calls, repeats,
                row.median, row.lo, row.hi, row.energy, row.kernel.c_str(),
                static_cast<unsigned long long>(row.force_calls),
                static_cast<unsigned long long>(row.pairs),
                static_cast<unsigned long long>(row.tile_bytes), row.threads_used,
                row.median * static_cast<double>(row.threads_used));
  }
  return 0;
}
