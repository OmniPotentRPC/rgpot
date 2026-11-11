// MIT License
// Copyright 2023--present rgpot developers
//
// Verlet skin. A move inside half the skin keeps the candidate index and
// still refreshes r^2. A move past half the skin rebuilds.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdio>

int main() {
  double pos[] = {0, 0, 0, 1.2, 0, 0, 0, 1.1, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  const std::array<bool, 3> periodic{{true, true, true}};
  rgpot::nlist::Neighbours cells;
  auto sum = [&](const double *xyz) {
    double total = 0;
    const auto step = rgpot::nlist::each(cells, xyz, 3, box, 2.0, periodic, 1.0);
    step([&](std::int32_t, std::int32_t, double, double, double, double r2) { total += r2; });
    return total;
  };
  const double first = sum(pos);
  if (cells.reused() || !(first > 0)) {
    std::fprintf(stderr, "first reused %d sum %.16g\n", cells.reused() ? 1 : 0, first);
    return 1;
  }
  pos[0] = 0.1;
  const double nudged = sum(pos);
  if (!cells.reused() || std::fabs(nudged - first) < 1e-12) {
    std::fprintf(stderr, "nudge reused %d %.16g %.16g\n", cells.reused() ? 1 : 0, first, nudged);
    return 1;
  }
  pos[0] = 1.0;
  const double jumped = sum(pos);
  if (cells.reused()) {
    std::fprintf(stderr, "jump reused the index, sum %.16g\n", jumped);
    return 1;
  }
  return 0;
}
