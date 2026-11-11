// Copyright (c) rgpot developers
//
// Times rgpot::build_system_edges, the list UmaPot hands to the graph.
// Molecular rows are every ordered pair. Periodic rows are the vesin
// list at a 6 angstrom cutoff, trimmed at 300 neighbors.
// Each sample is 11 calls. The report gives the minimum, the median,
// and the interquartile range of those calls.
// CppCore/meson.build registers the binary with benchmark().
// That target stays out of meson test and runs one at a time.

#include "rgpot/UmaPot/UmaEdges.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kCutoff = 6.0;
constexpr double kSpacing = 2.5;
constexpr int kRepeats = 11;
constexpr int kMaxNeighbors = 300;

struct Stats {
  double min = 0;
  double q1 = 0;
  double median = 0;
  double q3 = 0;
  double iqr = 0;
};

Stats summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  Stats s;
  const std::size_t n = v.size();
  s.min = v.front();
  s.q1 = v[n / 4];
  s.median = v[n / 2];
  s.q3 = v[(3 * n) / 4];
  s.iqr = s.q3 - s.q1;
  return s;
}

std::string cpu_model() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("model name", 0) != 0)
      continue;
    const auto pos = line.find(':');
    if (pos == std::string::npos)
      continue;
    std::string value = line.substr(pos + 1);
    while (!value.empty() && value.front() == ' ')
      value.erase(value.begin());
    return value;
  }
  return "unknown";
}

int logical_cpus() {
  std::ifstream in("/proc/cpuinfo");
  int n = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("processor", 0) == 0)
      ++n;
  }
  return n;
}

std::vector<double> place(int n, double side, std::uint32_t seed) {
  std::vector<double> pos(static_cast<std::size_t>(n) * 3);
  std::uint32_t state = seed;
  for (int i = 0; i < n * 3; ++i) {
    state = state * 1664525u + 1013904223u;
    pos[static_cast<std::size_t>(i)] =
        (state >> 8) * (1.0 / 16777216.0) * side;
  }
  return pos;
}

double side_for(int n) { return std::cbrt(static_cast<double>(n)) * kSpacing; }

int cell_count(double side) {
  const int per = static_cast<int>(std::floor(side / kCutoff));
  const int cells = per > 0 ? per : 1;
  return cells * cells * cells;
}

struct Sample {
  Stats stats;
  std::size_t edges = 0;
};

Sample time_calls(int n, double side, double molecular_box, int threads,
                  bool reuse) {
  const auto pos = place(n, side, 7u);
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  std::vector<int> z(static_cast<std::size_t>(n), 1);
  rgpot::UmaContract contract;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(kRepeats));
  std::size_t edges = 0;
  rgpot::UmaEdgeSession kept;
  kept.threads = threads;
  auto once = [&](rgpot::UmaEdgeSession &session) {
    rgpot::ForceInput in{static_cast<std::size_t>(n), pos.data(), z.data(),
                         box};
    const auto list = rgpot::build_system_edges(
        in, kCutoff, kCutoff, molecular_box, kMaxNeighbors, contract,
        /*n_systems=*/1, session);
    edges = list.neighbor.size();
  };
  if (reuse) {
    once(kept);
    for (int i = 0; i < kRepeats; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      once(kept);
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  } else {
    for (int i = 0; i < kRepeats; ++i) {
      rgpot::UmaEdgeSession session;
      session.threads = threads;
      const auto t0 = std::chrono::steady_clock::now();
      once(session);
      const auto t1 = std::chrono::steady_clock::now();
      samples.push_back(
          std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  return {summarize(samples), edges};
}

void report(const char *path, int n, int threads, double side, const Sample &s) {
  std::printf(
      "%s n=%d threads=%d median_us=%.3f min_us=%.3f iqr_us=%.3f edges=%zu cells=%d\n",
      path, n, threads, s.stats.median, s.stats.min, s.stats.iqr, s.edges,
      cell_count(side));
}

double exponent(const std::vector<double> &x, const std::vector<double> &y) {
  const double n = static_cast<double>(x.size());
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double lx = std::log(x[i]);
    const double ly = std::log(y[i]);
    sx += lx;
    sy += ly;
    sxx += lx * lx;
    sxy += lx * ly;
  }
  const double den = n * sxx - sx * sx;
  if (den == 0.0)
    return 0.0;
  return (n * sxy - sx * sy) / den;
}

} // namespace

int main() {
  std::printf("cpu_model=%s\n", cpu_model().c_str());
  std::printf("logical_cpus=%d\n", logical_cpus());
  std::printf("hardware_concurrency=%u\n",
              std::thread::hardware_concurrency());
  std::printf("repeats=%d cutoff_A=%.1f spacing_A=%.1f max_neighbors=%d\n",
              kRepeats, kCutoff, kSpacing, kMaxNeighbors);
  std::printf("iqr is s[3n/4] minus s[n/4] on the sorted sample of 11\n");

  std::printf("molecular\n");
  const int mol_n[] = {32, 64, 128, 256};
  for (int n : mol_n) {
    const double side = side_for(n);
    for (int threads : {1, 4}) {
      const Sample s = time_calls(n, side, /*molecular_box=*/side, threads,
                                  /*reuse=*/false);
      report("molecular", n, threads, side, s);
    }
  }

  std::printf("periodic_build\n");
  const int weak_n[] = {1024, 4096, 16384, 65536};
  std::vector<double> weak_x;
  std::vector<double> weak_y;
  for (int n : weak_n) {
    const double side = side_for(n);
    const Sample s =
        time_calls(n, side, /*molecular_box=*/0.0, /*threads=*/1, false);
    report("periodic_build", n, 1, side, s);
    weak_x.push_back(static_cast<double>(n));
    weak_y.push_back(s.stats.median);
  }
  std::printf("weak_exponent=%.6f\n", exponent(weak_x, weak_y));

  std::printf("periodic_reuse\n");
  {
    constexpr int n = 8192;
    const double side = side_for(n);
    const Sample s =
        time_calls(n, side, 0.0, /*threads=*/1, /*reuse=*/true);
    report("periodic_reuse", n, 1, side, s);
  }

  std::printf("strong_scaling\n");
  std::vector<double> strong_x;
  std::vector<double> strong_y;
  {
    constexpr int n = 8192;
    const double side = side_for(n);
    std::printf("strong_cells=%d\n", cell_count(side));
    for (int threads : {1, 2, 4, 8}) {
      const Sample s = time_calls(n, side, 0.0, threads, false);
      report("strong", n, threads, side, s);
      strong_x.push_back(static_cast<double>(threads));
      strong_y.push_back(s.stats.median);
    }
  }
  std::printf("strong_exponent=%.6f\n", exponent(strong_x, strong_y));

  std::printf("small_box\n");
  {
    const double side = 2.5 * kCutoff;
    const int n = static_cast<int>(std::lround(std::pow(side / kSpacing, 3.0)));
    const Sample s = time_calls(n, side, 0.0, 1, false);
    report("small_box", n, 1, side, s);
  }
  return 0;
}
