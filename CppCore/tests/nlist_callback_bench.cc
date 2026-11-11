// MIT License
// Copyright 2023--present rgpot developers
//
// Wall time and peak resident set of the geometry callback against the
// cell walk. One path per process: walk, callback. Sizes are FCC cubes.

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

struct Geom {
  std::vector<double> pos;
  double box[9];
};

Geom fcc(int n, double lattice) {
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  Geom geo;
  geo.pos.reserve(static_cast<std::size_t>(n) * n * n * 4 * 3);
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &b : basis) {
          geo.pos.push_back((ix + b[0]) * lattice);
          geo.pos.push_back((iy + b[1]) * lattice);
          geo.pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double edge = n * lattice;
  geo.box[0] = edge;
  geo.box[1] = 0;
  geo.box[2] = 0;
  geo.box[3] = 0;
  geo.box[4] = edge;
  geo.box[5] = 0;
  geo.box[6] = 0;
  geo.box[7] = 0;
  geo.box[8] = edge;
  return geo;
}

double timeWalk(const Geom &geo, double cutoff, int repeats, double &acc) {
  const auto nat = geo.pos.size() / 3;
  const std::array<bool, 3> periodic{{true, true, true}};
  rgpot::nlist::Neighbours cells;
  auto once = [&]() {
    rgpot::nlist::updateCells(cells, geo.pos.data(), nat, geo.box, cutoff, 0.0,
                              rgpot::nlist::Image::Round, rgpot::nlist::Mode::Stream, periodic);
    double sum = 0;
    rgpot::nlist::visitHalf(cells, 0, static_cast<std::int64_t>(nat),
                            [&](std::int32_t, std::int32_t, double, double, double, double r2) {
                              sum += r2;
                            });
    acc = sum;
  };
  once();
  once();
  std::vector<double> samples;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double timeCallback(const Geom &geo, double cutoff, int repeats, double &acc) {
  const auto nat = geo.pos.size() / 3;
  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Half};
  auto once = [&]() {
    double sum = 0;
    rgpot::nlist::each(geo.pos.data(), nat, geo.box, query,
                       [&](std::int32_t, std::int32_t, int, int, int, double, double, double,
                           double r) { sum += r; });
    acc = sum;
  };
  once();
  once();
  std::vector<double> samples;
  for (int r = 0; r < repeats; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s walk|callback cube\n", argv[0]);
    return 2;
  }
  const std::string which = argv[1];
  const int cube = std::atoi(argv[2]);
  const double lattice = 3.615;
  const double cutoff = 5.0;
  const Geom geo = fcc(cube, lattice);
  double acc = 0;
  const double seconds = which == "walk" ? timeWalk(geo, cutoff, 7, acc)
                                         : timeCallback(geo, cutoff, 7, acc);
  std::printf("path %s atoms %zu median_s %.9f rss_kb %ld acc %.6f\n", which.c_str(),
              geo.pos.size() / 3, seconds, peakRssKb(), acc);
  return which == "walk" || which == "callback" ? 0 : 2;
}
