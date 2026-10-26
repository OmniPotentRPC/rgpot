// MIT License
// Copyright 2023--present rgpot developers
//
// binAtoms packs every atom into a cell. The counts have to add up.

#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/nlist/HalfShellDetail.hpp"

namespace rgpot::nlist::detail {

bool makeGrid(const double *box, double edge, Grid &g);

void binAtoms(const double *pos, std::size_t n, const Grid &g,
              std::vector<int> &cell_of, std::vector<int> &rank,
              std::vector<int> &counts, std::vector<int> &start,
              std::vector<int> &atoms, bool ranks);

} // namespace rgpot::nlist::detail

TEST_CASE("half-shell bins hold every atom", "[HalfShell]") {
  const double box[9] = {12, 0, 0, 0, 12, 0, 0, 0, 12};
  const double pos[] = {0.1, 0.2,  0.3,  4.1,  0.2, 0.3, 8.5, 4.5,
                        8.5, 11.9, 11.9, 11.9, 0.0, 6.0, 3.0};
  const std::size_t n = 5;
  rgpot::nlist::detail::Grid grid;
  REQUIRE(rgpot::nlist::detail::makeGrid(box, 4.0, grid));
  REQUIRE(grid.nx == 3);
  REQUIRE(grid.ny == 3);
  REQUIRE(grid.nz == 3);
  REQUIRE(grid.ncell == 27);
  std::vector<int> cell_of;
  std::vector<int> rank;
  std::vector<int> counts;
  std::vector<int> start;
  std::vector<int> atoms;
  rgpot::nlist::detail::binAtoms(pos, n, grid, cell_of, rank, counts, start,
                                 atoms, true);
  REQUIRE(start.back() == static_cast<int>(n));
  int sum = 0;
  std::vector<int> seen(n, 0);
  for (int c = 0; c < grid.ncell; ++c) {
    const int a = start[static_cast<std::size_t>(c)];
    const int b = start[static_cast<std::size_t>(c) + 1];
    sum += b - a;
    for (int p = a; p < b; ++p) {
      const int atom = atoms[static_cast<std::size_t>(p)];
      REQUIRE(atom >= 0);
      REQUIRE(atom < static_cast<int>(n));
      REQUIRE(cell_of[static_cast<std::size_t>(atom)] == c);
      seen[static_cast<std::size_t>(atom)] += 1;
    }
  }
  REQUIRE(sum == static_cast<int>(n));
  for (int hits : seen) {
    REQUIRE(hits == 1);
  }
  REQUIRE(rank.size() == n);

  const double short_box[9] = {6, 0, 0, 0, 12, 0, 0, 0, 12};
  rgpot::nlist::detail::Grid too_small;
  REQUIRE_FALSE(rgpot::nlist::detail::makeGrid(short_box, 4.0, too_small));
}
