// MIT License
// Copyright 2023--present rgpot developers
//
// Stored half-list walk against each().team at 1M atoms. Both sum r^2.
// One path and one thread count per process.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "rgpot/nlist/NeighbourList.hpp"

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

std::vector<double> fcc(int n, double lattice, double &edge) {
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  pos.reserve(static_cast<std::size_t>(n) * n * n * 4 * 3);
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * lattice);
          pos.push_back((iy + b[1]) * lattice);
          pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  edge = n * lattice;
  return pos;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s stored|each threads\n", argv[0]);
    return 2;
  }
  const std::string which = argv[1];
  const int threads = std::atoi(argv[2]);
  const double lattice = 3.615;
  const double cutoff = 5.0;
  double edge = 0;
  const std::vector<double> pos = fcc(64, lattice, edge);
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  const std::array<bool, 3> periodic{{true, true, true}};
  const auto nat = pos.size() / 3;
  rgpot::nlist::Neighbours cells;
  std::vector<double> part(static_cast<std::size_t>(threads), 0.0);
  auto once = [&]() {
    std::fill(part.begin(), part.end(), 0.0);
    if (which == "stored") {
      rgpot::nlist::updateCells(cells, pos.data(), nat, box, cutoff, 0.0, rgpot::nlist::Image::Round,
                                rgpot::nlist::Mode::Stored, periodic);
      cells.forCellTeam(true, threads,
                        [&](int thread, std::int32_t, std::int32_t, double, double, double,
                            double r2) { part[static_cast<std::size_t>(thread)] += r2; });
    } else {
      rgpot::nlist::each(cells, pos.data(), nat, box, cutoff, periodic, 0.0)
          .team(threads)([&](int thread, std::int32_t, std::int32_t, double, double, double,
                             double r2) { part[static_cast<std::size_t>(thread)] += r2; });
    }
  };
  once();
  once();
  std::vector<double> samples;
  for (int r = 0; r < 7; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  double acc = 0;
  for (double value : part) {
    acc += value;
  }
  std::printf("path %s threads %d atoms %zu median_s %.9f rss_kb %ld acc %.6f\n", which.c_str(),
              threads, nat, samples[samples.size() / 2], peakRssKb(), acc);
  return which == "stored" || which == "each" ? 0 : 2;
}
