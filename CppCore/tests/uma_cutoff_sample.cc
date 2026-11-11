// Copyright 2023--present rgpot developers
//
// Writes one raw call per line for scripts/uma_cutoff_report.py.
// The process is pinned to the online CPUs. Each cell is 21 calls.

#include "rgpot/UmaPot/UmaEdges.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace {

constexpr double kCutoff = 6.0;
constexpr double kSpacing = 2.5;
constexpr int kCalls = 21;
constexpr int kMaxNeighbors = 300;

long rss_kb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) != 0)
      continue;
    return std::atol(line.c_str() + 6);
  }
  return 0;
}

void pin_process() {
  cpu_set_t set;
  CPU_ZERO(&set);
  const long n = sysconf(_SC_NPROCESSORS_ONLN);
  for (long cpu = 0; cpu < n; ++cpu)
    CPU_SET(static_cast<int>(cpu), &set);
  sched_setaffinity(0, sizeof(set), &set);
}

std::vector<double> place(int n, double side) {
  std::vector<double> pos(static_cast<std::size_t>(n) * 3);
  std::uint32_t state = 7u;
  for (int i = 0; i < n * 3; ++i) {
    state = state * 1664525u + 1013904223u;
    pos[static_cast<std::size_t>(i)] = (state >> 8) * (1.0 / 16777216.0) * side;
  }
  return pos;
}

double side_for(int n) { return std::cbrt(static_cast<double>(n)) * kSpacing; }

void emit_call(const char *series, unsigned seed, int reuse, int n, int threads, int call,
               double us) {
  std::printf(
      "series=%s seed=%u reuse=%d n=%d threads=%d call=%d us=%.3f rss_kb=%ld\n",
      series, seed, reuse, n, threads, call, us, rss_kb());
  std::fflush(stdout);
}

double time_one(const std::vector<double> &pos, const std::vector<int> &z, double side,
                int threads, rgpot::UmaEdgeSession &session) {
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  const auto t0 = std::chrono::steady_clock::now();
  const auto &list = rgpot::build_system_edges(in, kCutoff, kCutoff, 0.0, kMaxNeighbors,
                                               contract, 1, session);
  const auto t1 = std::chrono::steady_clock::now();
  if (list.neighbor.empty())
    std::fprintf(stderr, "empty edge list\n");
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

void oneshot(const char *series, unsigned seed, int n, int threads) {
  const double side = side_for(n);
  const auto pos = place(n, side);
  std::vector<int> z(static_cast<std::size_t>(n), 1);
  for (int call = 0; call < kCalls; ++call) {
    rgpot::UmaEdgeSession session;
    session.threads = threads;
    emit_call(series, seed, 0, n, threads, call, time_one(pos, z, side, threads, session));
  }
}

void reused(unsigned seed) {
  constexpr int n = 65536;
  constexpr int threads = 8;
  const double side = side_for(n);
  const auto pos = place(n, side);
  std::vector<int> z(static_cast<std::size_t>(n), 1);
  rgpot::UmaEdgeSession session;
  session.threads = threads;
  time_one(pos, z, side, threads, session);
  time_one(pos, z, side, threads, session);
  for (int leg = 0; leg < 2; ++leg) {
    for (int call = 0; call < kCalls; ++call) {
      emit_call("reused65536", seed, 1, n, threads, leg * kCalls + call,
                time_one(pos, z, side, threads, session));
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: uma_cutoff_sample seed\n");
    return 2;
  }
  const auto seed = static_cast<unsigned>(std::stoul(argv[1]));
  pin_process();
  reused(seed);
  const int sizes[] = {8192, 16384, 32768, 65536};
  for (int n : sizes)
    oneshot("weak_atoms", seed, n, n / 8192);
  for (int n : sizes)
    oneshot("by_size", seed, n, 1);
  return 0;
}
