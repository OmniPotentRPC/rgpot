// MIT License
// Copyright 2023--present rgpot developers
//
// The metatomic adapter copies five label columns and keeps the vectors
// in the table. This example does that copy without linking torch, and
// checks the labels of one pair.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  const double pos[] = {0.1, 0.2, 0.3, 1.2, 0.2, 0.3};
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.shape = rgpot::nlist::Pairs{rgpot::nlist::Shell::Full};
  query.cut = rgpot::nlist::Cut::Strict;
  query.fields.distances = false;
  rgpot::nlist::Table table;
  rgpot::nlist::build(table, pos, 2, box, query);
  const rgpot::nlist::View pairs = table.view();
  if (pairs.distance.size() != 0 || pairs.vector.size() != pairs.first.size() * 3) {
    std::fprintf(stderr, "field selection did not match the request\n");
    return 1;
  }
  std::vector<std::int32_t> labels(pairs.first.size() * 5);
  for (std::size_t k = 0; k < pairs.first.size(); ++k) {
    labels[5 * k] = pairs.first[k];
    labels[5 * k + 1] = pairs.second[k];
    labels[5 * k + 2] = pairs.shift[3 * k];
    labels[5 * k + 3] = pairs.shift[3 * k + 1];
    labels[5 * k + 4] = pairs.shift[3 * k + 2];
  }
  bool saw = false;
  for (std::size_t k = 0; k < pairs.first.size(); ++k) {
    if (labels[5 * k] == 0 && labels[5 * k + 1] == 1 && labels[5 * k + 2] == 0 &&
        labels[5 * k + 3] == 0 && labels[5 * k + 4] == 0) {
      const double dx = pairs.vector[3 * k] - (pos[3] - pos[0]);
      if (std::abs(dx) > 1e-12) {
        std::fprintf(stderr, "vector was copied out of the table\n");
        return 1;
      }
      saw = true;
    }
  }
  if (!saw) {
    std::fprintf(stderr, "missing the direct pair\n");
    return 1;
  }
  return 0;
}
