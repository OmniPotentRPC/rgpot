// Copyright (c) rgpot developers
//
// Microseconds per call for a cutoff neighbor list and for every
// ordered pair, across atom counts, and for 1, 2, 4, and 8 threads.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "rgpot/MetatomicPot/vesin_compat.hpp"
#include "vesin.h"

namespace {

constexpr double kCutoff = 6.0;

struct Sample {
  double us = 0.0;
  std::size_t edges = 0;
};

std::vector<double> place(int n, double side, std::uint32_t seed) {
  std::vector<double> pos(static_cast<std::size_t>(n) * 3);
  std::uint32_t state = seed;
  for (int i = 0; i < n * 3; ++i) {
    state = state * 1664525u + 1013904223u;
    pos[static_cast<std::size_t>(i)] = (state >> 8) * (1.0 / 16777216.0) * side;
  }
  return pos;
}

double side_for(int n) {
  const double reach = std::cbrt(static_cast<double>(n)) * 2.5;
  return reach > 14.0 ? reach : 14.0;
}

Sample time_cutoff(const std::vector<double> &pos, int n, double side, int threads) {
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  bool pbc[3] = {true, true, true};
  VesinOptions options{};
  rgpot::vesin_compat::fill_neighbor_options(options, kCutoff, true);
  rgpot::vesin_compat::set_n_threads(options, threads);
  const auto cpu = rgpot::vesin_compat::make_cpu_device();
  auto once = [&](std::size_t *edges) {
    VesinNeighborList list{};
    const char *err = nullptr;
    const int status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(pos.data()), static_cast<std::size_t>(n),
        reinterpret_cast<const double(*)[3]>(box), pbc, cpu, options, &list, &err);
    if (status != 0) {
      std::fprintf(stderr, "vesin failed: %s\n", err ? err : "");
      vesin_free(&list);
      return false;
    }
    if (edges)
      *edges = list.length;
    vesin_free(&list);
    return true;
  };
  for (int i = 0; i < 3; ++i)
    once(nullptr);
  std::size_t edges = 0;
  const auto t0 = std::chrono::steady_clock::now();
  int calls = 0;
  do {
    if (!once(&edges))
      return {};
    ++calls;
  } while (calls < 20 ||
           std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(40));
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / calls;
  return {us, edges};
}

Sample time_full(const std::vector<double> &pos, int n) {
  auto once = [&](std::size_t *edges) {
    std::size_t count = 0;
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j)
          continue;
        const double dx = pos[3 * j] - pos[3 * i];
        const double dy = pos[3 * j + 1] - pos[3 * i + 1];
        const double dz = pos[3 * j + 2] - pos[3 * i + 2];
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 == r2)
          ++count;
      }
    }
    if (edges)
      *edges = count;
  };
  for (int i = 0; i < 3; ++i)
    once(nullptr);
  std::size_t edges = 0;
  const auto t0 = std::chrono::steady_clock::now();
  int calls = 0;
  do {
    once(&edges);
    ++calls;
  } while (calls < 20 ||
           std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(40));
  const auto t1 = std::chrono::steady_clock::now();
  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / calls;
  return {us, edges};
}

} // namespace

int main() {
  std::printf("us_per_call cutoff_A=%.1f\n", kCutoff);
  std::printf("n threads cutoff_us nocutoff_us cutoff_edges full_edges\n");
  const int sizes[] = {16, 32, 64, 128, 256, 512};
  for (int n : sizes) {
    const double side = side_for(n);
    const auto pos = place(n, side, 7u);
    const Sample cut = time_cutoff(pos, n, side, 1);
    const Sample full = time_full(pos, n);
    std::printf("n=%d threads=1 cutoff_us=%.3f nocutoff_us=%.3f cutoff_edges=%zu full_edges=%zu\n",
                n, cut.us, full.us, cut.edges, full.edges);
  }
  std::printf("thread_scaling n=256\n");
  {
    constexpr int n = 256;
    const double side = side_for(n);
    const auto pos = place(n, side, 7u);
    const Sample full = time_full(pos, n);
    const int threads[] = {1, 2, 4, 8};
    for (int t : threads) {
      const Sample cut = time_cutoff(pos, n, side, t);
      std::printf("n=%d threads=%d cutoff_us=%.3f nocutoff_us=%.3f cutoff_edges=%zu full_edges=%zu\n",
                  n, t, cut.us, full.us, cut.edges, full.edges);
    }
  }
  return 0;
}
