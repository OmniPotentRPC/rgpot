// Copyright 2023--present rgpot developers
//
// One timed call of build_system_edges. The caller pins the process
// and records the load. Arguments: series n threads reuse seed.

#include "rgpot/UmaPot/UmaEdges.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace {

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

double time_call(const std::vector<double> &pos, const std::vector<int> &z, double side,
                 int threads, rgpot::UmaEdgeSession &session) {
  double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  rgpot::UmaContract contract;
  rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  const auto t0 = std::chrono::steady_clock::now();
  const auto &list = rgpot::build_system_edges(in, 6.0, 6.0, 0.0, 300, contract, 1, session);
  const auto t1 = std::chrono::steady_clock::now();
  if (list.neighbor.empty())
    std::fprintf(stderr, "empty\n");
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

double once(const std::vector<double> &pos, const std::vector<int> &z, double side,
            int threads, bool reuse) {
  if (!reuse) {
    rgpot::UmaEdgeSession warm;
    warm.threads = threads;
    time_call(pos, z, side, threads, warm);
    rgpot::UmaEdgeSession session;
    session.threads = threads;
    return time_call(pos, z, side, threads, session);
  }
  rgpot::UmaEdgeSession session;
  session.threads = threads;
  time_call(pos, z, side, threads, session);
  time_call(pos, z, side, threads, session);
  return time_call(pos, z, side, threads, session);
}

} // namespace

void emit(const char *series, unsigned seed, int reuse, int n, int threads, int call,
          double us) {
  std::printf("series=%s seed=%u reuse=%d n=%d threads=%d call=%d us=%.3f rss_kb=0\n",
              series, seed, reuse, n, threads, call, us);
  std::fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: uma_cutoff_abba matrix seed | reused-batch seed count\n");
    return 2;
  }
  pin_process();
  const std::string mode = argv[1];
  const unsigned seed = static_cast<unsigned>(std::stoul(argv[2]));
  if (mode == "reused-batch") {
    const int count = std::stoi(argv[3]);
    constexpr int n = 65536;
    const double side = std::cbrt(static_cast<double>(n)) * 2.5;
    const auto pos = place(n, side);
    std::vector<int> z(n, 1);
    for (int call = 0; call < count; ++call)
      emit("reused65536", seed, 1, n, 8, call, once(pos, z, side, 8, true));
    return 0;
  }
  const int spec[][2] = {{8192, 1}, {16384, 2}, {32768, 4}, {65536, 8}};
  int call = 0;
  for (int round = 0; round < 12; ++round) {
    for (int dir = 0; dir < 2; ++dir) {
      for (int i = 0; i < 4; ++i) {
        const int slot = dir == 0 ? i : 3 - i;
        const int n = spec[slot][0];
        const int threads = spec[slot][1];
        const double side = std::cbrt(static_cast<double>(n)) * 2.5;
        const auto pos = place(n, side);
        std::vector<int> z(static_cast<std::size_t>(n), 1);
        emit("weak_atoms", seed, 0, n, threads, call, once(pos, z, side, threads, false));
        ++call;
      }
    }
    constexpr int n = 65536;
    const double side = std::cbrt(static_cast<double>(n)) * 2.5;
    const auto pos = place(n, side);
    std::vector<int> z(n, 1);
    emit("reused65536", seed, 1, n, 8, round, once(pos, z, side, 8, true));
  }
  return 0;
}
