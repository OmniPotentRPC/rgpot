// MIT License
// Copyright 2023--present rgpot developers
//
// One neighbour-query timing. The Python driver in neighbour_api_bench.py
// owns the structure definitions and the other libraries.

#include <algorithm>
#include <chrono>
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

struct Geometry {
  std::vector<double> pos;
  double box[9]{};
};

Geometry fcc(int n, double lattice) {
  Geometry out;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  out.pos.reserve(static_cast<std::size_t>(n) * n * n * 4 * 3);
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &b : basis) {
          out.pos.push_back((ix + b[0]) * lattice);
          out.pos.push_back((iy + b[1]) * lattice);
          out.pos.push_back((iz + b[2]) * lattice);
        }
      }
    }
  }
  const double length = n * lattice;
  out.box[0] = length;
  out.box[4] = length;
  out.box[8] = length;
  return out;
}

Geometry triclinic() {
  Geometry out;
  const double cell[3][3] = {{6.0, 0.0, 0.0}, {1.2, 5.4, 0.0}, {0.3, 0.7, 5.8}};
  for (int i = 0; i < 3; ++i) {
    out.box[i] = cell[0][i];
    out.box[3 + i] = cell[1][i];
    out.box[6 + i] = cell[2][i];
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        const double f[3] = {(a + 0.2) / 4.0, (b + 0.2) / 4.0, (c + 0.2) / 4.0};
        for (int k = 0; k < 3; ++k) {
          out.pos.push_back(f[0] * cell[0][k] + f[1] * cell[1][k] + f[2] * cell[2][k]);
        }
      }
    }
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: neighbour_api_rgpot system format threads repeats\n");
    return 2;
  }
  const std::string system = argv[1];
  const std::string format = argv[2];
  const int threads = std::atoi(argv[3]);
  const int repeats = std::atoi(argv[4]);
  const double cutoff = 5.0;
  const double lattice = 3.615;
  Geometry geo = system == "tri" ? triclinic() : fcc(system == "cu1m" ? 64 : 30, lattice);
  rgpot::nlist::Query query;
  query.cutoff = cutoff;
  query.skin = 0;
  query.format = format == "full" ? rgpot::nlist::Format::Full : rgpot::nlist::Format::Half;
  query.cut = rgpot::nlist::Cut::Strict;
  query.fields = {true, true, true};
  query.mode = rgpot::nlist::Mode::Stored;
  query.periodic = {true, true, true};
  if (threads > 0) {
    setenv("OMP_NUM_THREADS", std::to_string(threads).c_str(), 1);
  }
  const std::size_t n = geo.pos.size() / 3;
  rgpot::nlist::Table table;
  for (int i = 0; i < 2; ++i) {
    rgpot::nlist::build(table, geo.pos.data(), n, geo.box, query);
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  for (int i = 0; i < repeats; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    rgpot::nlist::build(table, geo.pos.data(), n, geo.box, query);
    const auto t1 = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  std::printf("rgpot %s %s T=%d n=%zu pairs=%zu median %.6f rss_kb %ld\n", system.c_str(),
              format.c_str(), threads, n, table.size(), samples[samples.size() / 2], peakRssKb());
  return table.size() == 0 ? 1 : 0;
}
