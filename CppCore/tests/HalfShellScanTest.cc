// MIT License
// Copyright 2023--present rgpot developers
//
// The half-shell scan keeps every diamond bond and drops the rest.

#include <cmath>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/HalfShellDetail.hpp"

TEST_CASE("half-shell scan keeps the diamond bonds", "[HalfShell]") {
  const double cut = 1.8 * 2.0951;
  const double u = 1.1224620482975474;
  const double a0 = u * 2.0951 * 4.0 / std::sqrt(3.0);
  const double basis[8][3] = {{0, 0, 0},          {0.5, 0.5, 0},
                              {0.5, 0, 0.5},      {0, 0.5, 0.5},
                              {0.25, 0.25, 0.25}, {0.75, 0.75, 0.25},
                              {0.75, 0.25, 0.75}, {0.25, 0.75, 0.75}};
  std::vector<double> pos;
  for (int ix = 0; ix < 3; ++ix) {
    for (int iy = 0; iy < 3; ++iy) {
      for (int iz = 0; iz < 3; ++iz) {
        for (const auto &b : basis) {
          pos.push_back((ix + b[0]) * a0);
          pos.push_back((iy + b[1]) * a0);
          pos.push_back((iz + b[2]) * a0);
        }
      }
    }
  }
  const double box[9] = {3 * a0, 0, 0, 0, 3 * a0, 0, 0, 0, 3 * a0};
  rgpot::nlist::detail::Scratch scratch;
  rgpot::nlist::detail::fillPairs(pos.data(), box, pos.size() / 3, cut, cut,
                                  scratch);
  REQUIRE(scratch.lo.size() == 432);
  REQUIRE(scratch.lo.size() == scratch.hi.size());
  const double cut2 = cut * cut;
  for (std::size_t p = 0; p < scratch.lo.size(); ++p) {
    const double r2 = scratch.dx[p] * scratch.dx[p] +
                      scratch.dy[p] * scratch.dy[p] +
                      scratch.dz[p] * scratch.dz[p];
    REQUIRE(r2 > 0.0);
    REQUIRE(r2 < cut2);
    REQUIRE(scratch.lo[p] < scratch.hi[p]);
  }
}
