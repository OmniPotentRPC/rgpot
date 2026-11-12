// MIT License
// Copyright 2023--present rgpot developers
//
// Two systems in one call. Atoms in one system are not neighbours of
// the other. first and second index the concatenated positions.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cstdio>

int main() {
  const double positions[] = {0, 0, 0, 1.2, 0, 0, 10, 10, 10, 11.1, 10, 10};
  const std::int32_t count[] = {2, 2};
  const double box[] = {8, 0, 0, 0, 8, 0, 0, 0, 8, 8, 0, 0, 0, 8, 0, 0, 0, 8};
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  rgpot::nlist::Systems systems;
  systems.positions = positions;
  systems.count = count;
  systems.systems = 2;
  systems.box9 = box;
  rgpot::nlist::Batch batch;
  rgpot::nlist::build(batch, systems, query);
  if (batch.pairs.size() != 2 || batch.system.size() != 2) {
    std::fprintf(stderr, "batch pairs %zu\n", batch.pairs.size());
    return 1;
  }
  if (batch.system[0] != 0 || batch.system[1] != 1) {
    std::fprintf(stderr, "system ids %d %d\n", batch.system[0], batch.system[1]);
    return 1;
  }
  if (batch.pairs.first[1] < 2) {
    std::fprintf(stderr, "second system was indexed inside the first\n");
    return 1;
  }
  std::puts("batch-ok");
  return 0;
}
