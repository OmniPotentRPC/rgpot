// MIT License
// Copyright 2023--present rgpot developers
//
// A three-body term. The pair piece reads the half list. The angle reads
// the full list of the same sweep, so the bins are built once. A later
// call inside half the skin keeps that index and refreshes the vectors.

#include "rgpot/nlist/NeighbourList.hpp"

#include <cmath>
#include <cstdio>

int main() {
  const double pos[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  double moved[] = {0.05, 0, 0, 1, 0, 0, 0, 1, 0};
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  rgpot::nlist::Query query;
  query.cutoff = 1.5;
  query.skin = 0.4;
  query.cut = rgpot::nlist::Cut::Loose;
  rgpot::nlist::Sweep sweep;
  sweep.update(pos, 3, box, query);
  if (sweep.reused()) {
    std::fprintf(stderr, "first update reused the bins\n");
    return 1;
  }
  int half = 0;
  double pair = 0;
  sweep.each(rgpot::nlist::Shell::Half,
             [&](std::int32_t, std::int32_t, int, int, int, double, double, double, double r) {
               half += 1;
               pair += r;
             });
  int bonds = 0;
  double dot = 0;
  double vx[2] = {};
  double vy[2] = {};
  sweep.each(rgpot::nlist::Shell::Full,
             [&](std::int32_t i, std::int32_t, int, int, int, double x, double y, double, double) {
               if (i != 0 || bonds >= 2) {
                 return;
               }
               vx[bonds] = x;
               vy[bonds] = y;
               bonds += 1;
             });
  if (half != 3 || bonds != 2) {
    std::fprintf(stderr, "half %d bonds %d\n", half, bonds);
    return 1;
  }
  dot = vx[0] * vx[1] + vy[0] * vy[1];
  if (std::fabs(dot) > 1e-12) {
    std::fprintf(stderr, "angle %.16g\n", dot);
    return 1;
  }
  sweep.update(moved, 3, box, query);
  if (!sweep.reused()) {
    std::fprintf(stderr, "a move inside half the skin rebuilt\n");
    return 1;
  }
  double refreshed = 0;
  sweep.each(rgpot::nlist::Shell::Half,
             [&](std::int32_t i, std::int32_t j, int, int, int, double x, double, double, double) {
               if (i == 0 && j == 1) {
                 refreshed = x;
               }
             });
  if (std::fabs(refreshed - 0.95) > 1e-12) {
    std::fprintf(stderr, "vector %.16g\n", refreshed);
    return 1;
  }
  (void)pair;
  return 0;
}
