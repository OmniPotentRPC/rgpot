// MIT License
// Copyright 2023--present rgpot developers
//
// Per-thread accumulators on the cell team. Each thread owns the cells
// forCellTeam gives it and adds r^2 into its own slot. The sum matches
// one thread.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const double lattice = 3.615;
  const int n = 6;
  const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
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
  const double edge = n * lattice;
  const double box[9] = {edge, 0, 0, 0, edge, 0, 0, 0, edge};
  const std::array<bool, 3> periodic{{true, true, true}};
  const double cutoff = 5.0;
  const auto nat = pos.size() / 3;
  rgpot::nlist::Neighbours cells;
  double serial = 0;
  rgpot::nlist::each(cells, pos.data(), nat, box, cutoff, periodic)(
      [&](std::int32_t, std::int32_t, double, double, double, double r2) { serial += r2; });
  double part[4] = {};
  rgpot::nlist::each(cells, pos.data(), nat, box, cutoff, periodic).team(4)(
      [&](int thread, std::int32_t, std::int32_t, double, double, double, double r2) {
        part[thread] += r2;
      });
  double parallel = 0;
  for (double value : part) {
    parallel += value;
  }
  if (std::fabs(parallel - serial) > 1e-6) {
    std::fprintf(stderr, "serial %.16g team %.16g\n", serial, parallel);
    return 1;
  }
  if (!(serial > 0)) {
    std::fprintf(stderr, "empty list\n");
    return 1;
  }
  return 0;
}
