// MIT License
// Copyright 2023--present rgpot developers
//
// Three-body angle from a full list. Atoms sit at (0,0,0), (1,0,0) and
// (0,1,0). The two vectors out of the origin are orthogonal.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const double pos[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  rgpot::nlist::Query query;
  query.cutoff = 1.5;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos, 3, box, query);
  const rgpot::nlist::View pairs = table.view();
  std::vector<double> vx;
  std::vector<double> vy;
  for (std::size_t k = 0; k < pairs.first.size(); ++k) {
    if (pairs.first[k] != 0) {
      continue;
    }
    vx.push_back(pairs.vector[3 * k]);
    vy.push_back(pairs.vector[3 * k + 1]);
  }
  if (vx.size() != 2) {
    std::fprintf(stderr, "origin has %zu neighbours\n", vx.size());
    return 1;
  }
  const double dot = vx[0] * vx[1] + vy[0] * vy[1];
  if (std::abs(dot) > 1e-12) {
    std::fprintf(stderr, "angle dot %.16g\n", dot);
    return 1;
  }
  return 0;
}
