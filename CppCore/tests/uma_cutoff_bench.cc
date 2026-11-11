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

struct Band {
  double median = 0;
  double lo = 0;
  double hi = 0;
};

double median_of(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  if (n % 2 == 1)
    return v[n / 2];
  return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

Band bootstrap_median(const std::vector<double> &samples, std::uint32_t seed) {
  Band band;
  band.median = median_of(samples);
  constexpr int kBoots = 10000;
  std::vector<double> boots;
  boots.reserve(kBoots);
  std::vector<double> draw(samples.size());
  std::uint32_t state = seed;
  for (int i = 0; i < kBoots; ++i) {
    for (std::size_t k = 0; k < samples.size(); ++k) {
      state = state * 1664525u + 1013904223u;
      draw[k] = samples[(state >> 8) % samples.size()];
    }
    boots.push_back(median_of(draw));
  }
  std::sort(boots.begin(), boots.end());
  band.lo = boots[250];
  band.hi = boots[9749];
  return band;
}

Band bootstrap_slope(const std::vector<double> &left, double left_x,
                     const std::vector<double> &right, double right_x,
                     std::uint32_t seed) {
  Band band;
  const double point =
      std::log(median_of(right) / median_of(left)) / std::log(right_x / left_x);
  band.median = point;
  constexpr int kBoots = 10000;
  std::vector<double> boots;
  boots.reserve(kBoots);
  std::uint32_t state = seed;
  auto pick = [&](const std::vector<double> &src) {
    state = state * 1664525u + 1013904223u;
    return src[(state >> 8) % src.size()];
  };
  for (int i = 0; i < kBoots; ++i) {
    std::vector<double> a(left.size());
    std::vector<double> b(right.size());
    for (double &v : a)
      v = pick(left);
    for (double &v : b)
      v = pick(right);
    const double slope = std::log(median_of(b) / median_of(a)) / std::log(right_x / left_x);
    boots.push_back(slope);
  }
  std::sort(boots.begin(), boots.end());
  band.lo = boots[250];
  band.hi = boots[9749];
  return band;
}

double time_one(const std::vector<double> &pos, const std::vector<int> &z, double side,
                int threads, int policy, rgpot::UmaEdgeSession &session) {
  session.threads = threads;
  session.skin_policy = policy;
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  const auto t0 = std::chrono::steady_clock::now();
  const auto list = rgpot::build_system_edges(in, kCutoff, kCutoff, 0.0, kMaxNeighbors,
                                              contract, 1, session);
  const auto t1 = std::chrono::steady_clock::now();
  if (list.neighbor.empty())
    std::fprintf(stderr, "empty edge list\n");
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

void warm_session(const std::vector<double> &pos, const std::vector<int> &z, double side,
                  int threads, int policy, rgpot::UmaEdgeSession &session) {
  const int warmup = policy == 0 ? 2 : 1;
  for (int i = 0; i < warmup; ++i)
    time_one(pos, z, side, threads, policy, session);
}

int run_abba() {
  constexpr int kLeg = 21;
  // Ends are skin 0, then v1.6, with the tip in the center, then the reverse.
  const int policies[] = {1, 0, 2, 2, 0, 1};
  const char *names[] = {"", "skin0", "tip", "v16"};
  const int sizes[] = {8192, 65536};
  const int thread_choices[] = {1, static_cast<int>(std::thread::hardware_concurrency())};
  std::printf("cpu_model=%s\n", cpu_model().c_str());
  std::printf("logical_cpus=%d\n", logical_cpus());
  std::printf("hardware_concurrency=%u\n", std::thread::hardware_concurrency());
  std::printf("design=ABBA legs=skin0,v16,tip,tip,v16,skin0 calls_per_leg=%d\n", kLeg);
  std::printf("ci=bootstrap median percentile 2.5 and 97.5 over 10000 resamples\n");
  std::printf("policy 0 is v1.6, policy 1 is skin 0, policy 2 is the previous tip\n");

  std::vector<double> weak_left;
  std::vector<double> weak_right;
  std::vector<double> strong_left;
  std::vector<double> strong_right;

  for (int reuse = 0; reuse < 2; ++reuse) {
    const char *mode = reuse ? "reused" : "oneshot";
    for (int n : sizes) {
      const double side = side_for(n);
      const auto pos = place(n, side, 7u);
      std::vector<int> z(static_cast<std::size_t>(n), 1);
      for (int threads : thread_choices) {
        std::vector<double> samples[3];
        int leg_i = 0;
        for (int policy : policies) {
          const std::size_t before = samples[policy].size();
          if (reuse) {
            rgpot::UmaEdgeSession session;
            warm_session(pos, z, side, threads, policy, session);
            for (int i = 0; i < kLeg; ++i)
              samples[policy].push_back(time_one(pos, z, side, threads, policy, session));
          } else {
            for (int i = 0; i < kLeg; ++i) {
              rgpot::UmaEdgeSession session;
              samples[policy].push_back(time_one(pos, z, side, threads, policy, session));
            }
          }
          const double leg_median = median_of(std::vector<double>(
              samples[policy].begin() + static_cast<std::ptrdiff_t>(before), samples[policy].end()));
          std::printf("leg %s n=%d threads=%d reuse=%d policy=%s median_us=%.3f\n", mode, n,
                      threads, reuse, policy == 0 ? "v16" : names[policy], leg_median);
          std::fflush(stdout);
          ++leg_i;
          (void)leg_i;
        }
        for (int policy : {1, 2, 0}) {
          const Band band = bootstrap_median(samples[policy], 20261110u + policy + n + threads);
          std::printf(
              "%s n=%d threads=%d policy=%s median_us=%.3f ci95_lo=%.3f ci95_hi=%.3f n_calls=%zu\n",
              mode, n, threads, policy == 0 ? "v16" : names[policy], band.median, band.lo,
              band.hi, samples[policy].size());
        }
        if (!reuse && threads == 1) {
          if (n == 8192)
            weak_left = samples[0];
          if (n == 65536)
            weak_right = samples[0];
        }
        if (!reuse && n == 8192) {
          if (threads == 1)
            strong_left = samples[0];
          else
            strong_right = samples[0];
        }
      }
    }
  }

  const Band weak = bootstrap_slope(weak_left, 8192, weak_right, 65536, 20261111u);
  const Band strong =
      bootstrap_slope(strong_left, 1, strong_right,
                      static_cast<double>(std::thread::hardware_concurrency()), 20261112u);
  std::printf("exponent_host=%s logical_cpus=%d\n", cpu_model().c_str(), logical_cpus());
  std::printf(
      "weak_exponent=%.6f ci95_lo=%.6f ci95_hi=%.6f sample=oneshot v16 n=8192,65536 threads=1\n",
      weak.median, weak.lo, weak.hi);
  std::printf(
      "strong_exponent=%.6f ci95_lo=%.6f ci95_hi=%.6f sample=oneshot v16 n=8192 threads=1,%d\n",
      strong.median, strong.lo, strong.hi,
      static_cast<int>(std::thread::hardware_concurrency()));
  return 0;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--abba")
    return run_abba();
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
