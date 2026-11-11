// MIT License
// Copyright 2023--present rgpot developers
//
// Metatomic labels and the distance vector from one sweep. The vector is
// r_j - r_i + S H. The five label columns are first, second and the shift.

#include "rgpot/nlist/adapt.hpp"

#include <cmath>
#include <cstdio>

int main() {
  const double pos[] = {0.1, 0.2, 0.3, 1.4, 0.2, 0.3};
  const double box[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
  const bool periodic[3] = {true, true, true};
  rgpot::nlist::Query query = rgpot::nlist::metatomicQuery(2.0, true, true, periodic);
  rgpot::nlist::Sweep sweep;
  sweep.update(pos, 2, box, query);
  bool saw = false;
  rgpot::nlist::metatomicPairs(
      sweep, true, [&](std::int32_t i, std::int32_t j, int sa, int sb, int sc, double x, double y,
                       double z) {
        if (i == 0 && j == 1 && sa == 0 && sb == 0 && sc == 0) {
          const double dx = x - (pos[3] - pos[0]);
          const double dy = y - (pos[4] - pos[1]);
          const double dz = z - (pos[5] - pos[2]);
          if (std::fabs(dx) > 1e-12 || std::fabs(dy) > 1e-12 || std::fabs(dz) > 1e-12) {
            std::fprintf(stderr, "vector %.16g %.16g %.16g\n", dx, dy, dz);
            saw = false;
            return;
          }
          saw = true;
        }
      });
  if (!saw) {
    std::fprintf(stderr, "missing the direct pair\n");
    return 1;
  }
  return 0;
}
