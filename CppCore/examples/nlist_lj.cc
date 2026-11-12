// MIT License
// Copyright 2023--present rgpot developers
//
// Half-list Lennard-Jones dimer at the potential minimum. The default
// query is a half shell, a strict cutoff and periodic boundaries. The
// cell is large, so the only pair is the two atoms. Energy is -1.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdio>

int main() {
  const double r = std::pow(2.0, 1.0 / 6.0);
  const double pos[] = {0, 0, 0, r, 0, 0};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  rgpot::nlist::Query query;
  query.cutoff = 2.5;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos, 2, box, query);
  const rgpot::nlist::View pairs = table.view();
  if (pairs.first.size() != 1 || pairs.distance.size() != 1) {
    std::fprintf(stderr, "expected one pair\n");
    return 1;
  }
  const double sr = 1.0 / pairs.distance[0];
  const double sr2 = sr * sr;
  const double sr6 = sr2 * sr2 * sr2;
  const double energy = 4.0 * (sr6 * sr6 - sr6);
  if (std::abs(energy + 1.0) > 1e-12) {
    std::fprintf(stderr, "energy %.16g\n", energy);
    return 1;
  }
  if (pairs.first[0] != 0 || pairs.second[0] != 1) {
    std::fprintf(stderr, "half list kept the wrong direction\n");
    return 1;
  }
  return 0;
}
