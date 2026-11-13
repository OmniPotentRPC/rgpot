// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time of EAM energy and forces. Not a correctness test.
// One thread is the default. OMP_NUM_THREADS selects the strong and weak
// scaling rows. Medians are over seven warm calls after two discarded
// calls, with the neighbour list reused.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#ifndef EAM_STANDALONE
#define EAM_STANDALONE
#endif
#include "rgpot/EAM/EAM.h"

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

std::string cpuModel() {
  std::ifstream in("/proc/cpuinfo");
  std::string line;
  while (std::getline(in, line)) {
    const std::string key = "model name";
    if (line.compare(0, key.size(), key) != 0) {
      continue;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string name = line.substr(colon + 1);
    const auto first = name.find_first_not_of(" \t");
    if (first != std::string::npos) {
      name.erase(0, first);
    }
    return name;
  }
  return "unknown";
}

std::vector<double> fcc(int nx, int ny, int nz, double a, double *box) {
  const double basis[4][3] = {
      {0.0, 0.0, 0.0},
      {0.0, 0.5, 0.5},
      {0.5, 0.0, 0.5},
      {0.5, 0.5, 0.0},
  };
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(nx * ny * nz * 4 * 3));
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a);
          pos.push_back((iy + b[1]) * a);
          pos.push_back((iz + b[2]) * a);
        }
      }
    }
  }
  box[0] = nx * a;
  box[1] = 0.0;
  box[2] = 0.0;
  box[3] = 0.0;
  box[4] = ny * a;
  box[5] = 0.0;
  box[6] = 0.0;
  box[7] = 0.0;
  box[8] = nz * a;
  return pos;
}

double medianWarm(EAM &pot, int n, const double *pos, const int *z,
                  const double *box, int repeats) {
  std::vector<double> force(static_cast<std::size_t>(n) * 3);
  double energy = 0.0;
  double variance = 0.0;
  pot.force(n, pos, z, force.data(), &energy, &variance, box);
  pot.force(n, pos, z, force.data(), &energy, &variance, box);
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    pot.force(n, pos, z, force.data(), &energy, &variance, box);
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main() {
  constexpr int kRepeats = 7;
  constexpr double kA = 4.05;
  std::printf("cpu: %s\n", cpuModel().c_str());
  std::printf("repeats: %d warm calls, median, list reused\n", kRepeats);
  std::printf("compiler flags for the reported run are the build's\n");

  setThreads("");
  std::printf("\nserial\n");
  std::printf("%8s %12s %12s\n", "atoms", "cold_s", "median_s");
  const int edge[] = {3, 5, 8, 12, 20, 30};
  for (int nrep : edge) {
    double box[9] = {};
    const std::vector<double> pos = fcc(nrep, nrep, nrep, kA, box);
    const int n = static_cast<int>(pos.size() / 3);
    std::vector<int> z(static_cast<std::size_t>(n), 13);
    EAM pot;
    std::vector<double> force(pos.size());
    double energy = 0.0;
    double variance = 0.0;
    const auto c0 = std::chrono::steady_clock::now();
    pot.force(n, pos.data(), z.data(), force.data(), &energy, &variance, box);
    const auto c1 = std::chrono::steady_clock::now();
    const double cold = std::chrono::duration<double>(c1 - c0).count();
    const double warm =
        medianWarm(pot, n, pos.data(), z.data(), box, kRepeats);
    std::printf("%8d %12.6f %12.6f\n", n, cold, warm);
  }

  double box[9] = {};
  const std::vector<double> strongPos = fcc(30, 30, 30, kA, box);
  const int strongN = static_cast<int>(strongPos.size() / 3);
  std::vector<int> strongZ(static_cast<std::size_t>(strongN), 13);
  std::printf("\nstrong n=%d\n", strongN);
  std::printf("%4s %12s\n", "T", "median_s");
  double serial = 0.0;
  for (const char *threads : {"1", "2", "4", "8"}) {
    setThreads(threads);
    EAM pot;
    const double warm = medianWarm(pot, strongN, strongPos.data(),
                                   strongZ.data(), box, kRepeats);
    if (threads[0] == '1' && threads[1] == '\0') {
      serial = warm;
    }
    const double speedup = serial > 0.0 ? serial / warm : 0.0;
    std::printf("%4s %12.6f  speedup %.3f\n", threads, warm, speedup);
  }

  std::printf("\nweak 4000 atoms per thread\n");
  std::printf("%4s %8s %12s\n", "T", "atoms", "median_s");
  const int shape[][3] = {
      {10, 10, 10},
      {10, 10, 20},
      {10, 20, 20},
      {20, 20, 20},
  };
  const char *threads[] = {"1", "2", "4", "8"};
  for (int i = 0; i < 4; ++i) {
    setThreads(threads[i]);
    double wbox[9] = {};
    const std::vector<double> pos =
        fcc(shape[i][0], shape[i][1], shape[i][2], kA, wbox);
    const int n = static_cast<int>(pos.size() / 3);
    std::vector<int> z(static_cast<std::size_t>(n), 13);
    EAM pot;
    const double warm =
        medianWarm(pot, n, pos.data(), z.data(), wbox, kRepeats);
    std::printf("%4s %8d %12.6f\n", threads[i], n, warm);
  }
  return 0;
}
