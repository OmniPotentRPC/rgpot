// Copyright 2023--present rgpot developers
//
// One process times three edge builders on the same calls.
// The order of the three versions rotates, so each one runs first
// on a third of the calls. CPUs 0 through 7 are pinned.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sched.h>
#include <unistd.h>

extern "C" {
double rgpot_time_v183(const double *pos, const int *atomic, int n, int threads, int reuse);
double rgpot_time_v184(const double *pos, const int *atomic, int n, int threads, int reuse);
double rgpot_time_v186(const double *pos, const int *atomic, int n, int threads, int reuse);
}

#ifndef RGPOT_V183_HEAD
#define RGPOT_V183_HEAD "unset"
#endif
#ifndef RGPOT_V183_TREE
#define RGPOT_V183_TREE "unset"
#endif
#ifndef RGPOT_V184_HEAD
#define RGPOT_V184_HEAD "unset"
#endif
#ifndef RGPOT_V184_TREE
#define RGPOT_V184_TREE "unset"
#endif
#ifndef RGPOT_V186_HEAD
#define RGPOT_V186_HEAD "unset"
#endif
#ifndef RGPOT_V186_TREE
#define RGPOT_V186_TREE "unset"
#endif

namespace {

void pin_cpus() {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu = 0; cpu < 8; ++cpu)
    CPU_SET(cpu, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::perror("sched_setaffinity");
    std::exit(1);
  }
}

std::string cpu_model() {
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
    while (*colon == ' ' || *colon == '\t')
      ++colon;
    model = colon;
    while (!model.empty() && (model.back() == '\n' || model.back() == ' '))
      model.pop_back();
    break;
  }
  std::fclose(file);
  for (char &ch : model) {
    if (ch == ' ')
      ch = '_';
  }
  return model;
}

std::string load_now() {
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
  while (!load.empty() && (load.back() == '\n' || load.back() == ' '))
    load.pop_back();
  return load;
}

std::string hostname_now() {
  char host[256];
  if (gethostname(host, sizeof host) != 0)
    return "unknown";
  host[sizeof host - 1] = '\0';
  return host;
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

struct Version {
  const char *name;
  const char *head;
  const char *tree;
  double (*time)(const double *, const int *, int, int, int);
};

void emit(const char *series, unsigned seed, int n, int threads, int call, const Version &ver,
          double us, const std::string &host, const std::string &cpu) {
  std::printf(
      "series=%s seed=%u reuse=0 n=%d threads=%d call=%d us=%.3f rss_kb=0 "
      "ver=%s head=%s tree=%s hostname=%s cpu=%s load=%s cpus=0-7\n",
      series, seed, n, threads, call, us, ver.name, ver.head, ver.tree, host.c_str(),
      cpu.c_str(), load_now().c_str());
  std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: uma_cutoff_panel seed\n");
    return 2;
  }
  pin_cpus();
  const unsigned seed = static_cast<unsigned>(std::stoul(argv[1]));
  const bool smoke = argc > 2 && std::strcmp(argv[2], "smoke") == 0;
  if (smoke) {
    const std::string host = hostname_now();
    const std::string cpu = cpu_model();
    const Version versions[3] = {
        {"v183", RGPOT_V183_HEAD, RGPOT_V183_TREE, rgpot_time_v183},
        {"v184", RGPOT_V184_HEAD, RGPOT_V184_TREE, rgpot_time_v184},
        {"v186", RGPOT_V186_HEAD, RGPOT_V186_TREE, rgpot_time_v186},
    };
    constexpr int n = 65536;
    const double side = std::cbrt(static_cast<double>(n)) * 2.5;
    const auto pos = place(n, side);
    std::vector<int> z(n, 1);
    for (int call = 0; call < 8; ++call) {
      for (int k = 0; k < 3; ++k) {
        const Version &ver = versions[(call + k) % 3];
        const double us = ver.time(pos.data(), z.data(), n, 8, 0);
        emit("weak", seed, n, 8, call, ver, us, host, cpu);
      }
    }
    return 0;
  }
  const std::string host = hostname_now();
  const std::string cpu = cpu_model();
  const Version versions[3] = {
      {"v183", RGPOT_V183_HEAD, RGPOT_V183_TREE, rgpot_time_v183},
      {"v184", RGPOT_V184_HEAD, RGPOT_V184_TREE, rgpot_time_v184},
      {"v186", RGPOT_V186_HEAD, RGPOT_V186_TREE, rgpot_time_v186},
  };
  const int weak[][2] = {{8192, 1}, {16384, 2}, {32768, 4}, {65536, 8}};
  const int strong[] = {1, 2, 4, 8};
  int call = 0;
  for (int round = 0; round < 12; ++round) {
    for (int dir = 0; dir < 2; ++dir) {
      for (int i = 0; i < 4; ++i) {
        const int slot = dir == 0 ? i : 3 - i;
        const int n = weak[slot][0];
        const int threads = weak[slot][1];
        const double side = std::cbrt(static_cast<double>(n)) * 2.5;
        const auto pos = place(n, side);
        std::vector<int> z(static_cast<std::size_t>(n), 1);
        for (int k = 0; k < 3; ++k) {
          const Version &ver = versions[(round + k) % 3];
          const double us = ver.time(pos.data(), z.data(), n, threads, 0);
          emit("weak", seed, n, threads, call, ver, us, host, cpu);
        }
        ++call;
      }
      constexpr int n = 65536;
      const double side = std::cbrt(static_cast<double>(n)) * 2.5;
      const auto pos = place(n, side);
      std::vector<int> z(n, 1);
      for (int threads : strong) {
        for (int k = 0; k < 3; ++k) {
          const Version &ver = versions[(round + k) % 3];
          const double us = ver.time(pos.data(), z.data(), n, threads, 0);
          emit("strong", seed, n, threads, call, ver, us, host, cpu);
        }
        ++call;
      }
    }
  }
  return 0;
}
