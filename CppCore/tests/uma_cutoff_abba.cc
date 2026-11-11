// Copyright 2023--present rgpot developers
//
// One timed call of build_system_edges. The caller pins the process
// and records the load. Arguments: series n threads reuse seed.

#include "rgpot/UmaPot/UmaEdges.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

namespace {

void pin_process() {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu = 0; cpu < 8; ++cpu)
    CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::perror("sched_setaffinity");
    std::exit(1);
  }
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

#ifndef RGPOT_ROW_HEAD
#define RGPOT_ROW_HEAD "unset"
#endif
#ifndef RGPOT_ROW_TREE
#define RGPOT_ROW_TREE "unset"
#endif

std::string row_host() {
  char host[256];
  if (gethostname(host, sizeof host) != 0)
    return "unknown";
  host[sizeof host - 1] = '\0';
  return host;
}

std::string row_cpu() {
  FILE *file = std::fopen("/proc/cpuinfo", "r");
  if (!file)
    return "unknown";
  char line[256];
  std::string model = "unknown";
  while (std::fgets(line, sizeof line, file)) {
    if (std::strncmp(line, "model name", 10) != 0)
      continue;
    const char *colon = std::strchr(line, ':');
    if (!colon)
      break;
    ++colon;
    while (*colon == ' ')
      ++colon;
    model = colon;
    while (!model.empty() && (model.back() == '\n' || model.back() == ' '))
      model.pop_back();
    break;
  }
  std::fclose(file);
  for (char &ch : model)
    if (ch == ' ')
      ch = '_';
  return model;
}

std::string row_load() {
  FILE *file = std::fopen("/proc/loadavg", "r");
  if (!file)
    return "0";
  char buf[64];
  if (!std::fgets(buf, sizeof buf, file))
    buf[0] = '0';
  std::fclose(file);
  std::string load = buf;
  const auto space = load.find(' ');
  if (space != std::string::npos)
    load.resize(space);
  while (!load.empty() && load.back() == '\n')
    load.pop_back();
  return load;
}

void emit(const char *series, unsigned seed, int reuse, int n, int threads, int call,
          double us) {
  std::printf(
      "series=%s seed=%u reuse=%d n=%d threads=%d call=%d us=%.3f rss_kb=0 "
      "head=%s tree=%s hostname=%s cpu=%s load=%s cpus=0-7\n",
      series, seed, reuse, n, threads, call, us, RGPOT_ROW_HEAD, RGPOT_ROW_TREE,
      row_host().c_str(), row_cpu().c_str(), row_load().c_str());
  std::fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: uma_cutoff_abba matrix seed | reused-batch seed count | "
                 "once series n threads reuse seed call\n");
    return 2;
  }
  pin_process();
  const std::string mode = argv[1];
  if (mode == "once") {
    const char *series = argv[2];
    const int n = std::stoi(argv[3]);
    const int threads = std::stoi(argv[4]);
    const bool reuse = std::stoi(argv[5]) != 0;
    const unsigned seed = static_cast<unsigned>(std::stoul(argv[6]));
    const int call = std::stoi(argv[7]);
    const double side = std::cbrt(static_cast<double>(n)) * 2.5;
    const auto pos = place(n, side);
    std::vector<int> z(static_cast<std::size_t>(n), 1);
    emit(series, seed, reuse ? 1 : 0, n, threads, call, once(pos, z, side, threads, reuse));
    return 0;
  }
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
