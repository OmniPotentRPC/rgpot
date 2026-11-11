// MIT License
// Copyright 2023--present rgpot developers
//
// EMT at one million FCC atoms. Median of seven calls after two warmups.
// One thread count per process.

#include "rgpot/EMT/EMTPot.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

long peakRssKb() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.compare(0, 6, "VmHWM:") == 0) {
      return std::strtol(line.c_str() + 6, nullptr, 10);
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const int cube = argc > 1 ? std::atoi(argv[1]) : 64;
  const int threads = argc > 2 ? std::atoi(argv[2]) : 1;
  const bool check = argc > 3 && std::string(argv[3]) == "check";
  const double lattice = 3.615;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> raw;
  for (int ix = 0; ix < cube; ++ix) {
    for (int iy = 0; iy < cube; ++iy) {
      for (int iz = 0; iz < cube; ++iz) {
        for (const auto &b : basis) {
          raw.push_back((ix + b[0]) * lattice);
          raw.push_back((iy + b[1]) * lattice);
          raw.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const auto nat = raw.size() / 3;
  rgpot::types::AtomMatrix positions(nat, 3);
  std::vector<int> types(nat, 29);
  for (std::size_t i = 0; i < nat; ++i) {
    positions(i, 0) = raw[3 * i];
    positions(i, 1) = raw[3 * i + 1];
    positions(i, 2) = raw[3 * i + 2];
  }
  const double edge = cube * lattice;
  const std::array<std::array<double, 3>, 3> box{{
      {edge, 0, 0},
      {0, edge, 0},
      {0, 0, edge},
  }};
  rgpot::EMTPot pot(rgpot::EMTConfig{.rasmussen = false});
  auto once = [&]() { return pot(positions, types, box); };
  if (check) {
    const auto result = once();
    double fsum = 0;
    const auto &forces = std::get<1>(result);
    for (std::size_t i = 0; i < nat; ++i) {
      fsum += forces(i, 0) + forces(i, 1) + forces(i, 2);
    }
    std::printf("check atoms %zu energy %.16e fsum %.16e\n", nat, std::get<0>(result), fsum);
    return 0;
  }
  once();
  once();
  std::vector<double> samples;
  double energy = 0;
  for (int r = 0; r < 7; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto result = once();
    const auto t1 = std::chrono::steady_clock::now();
    energy = std::get<0>(result);
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  std::printf("threads %d atoms %zu median_s %.9f rss_kb %ld energy %.16e\n", threads, nat,
              samples[samples.size() / 2], peakRssKb(), energy);
  return 0;
}
