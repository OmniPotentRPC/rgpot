#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Shared by the half-shell scan and the legacy-order sort. Not a public
// header: callers use HalfShell.hpp.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rgpot::nlist::detail {

struct Grid {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int ncell = 0;
  double lx = 0;
  double ly = 0;
  double lz = 0;
};

struct Key {
  int cell;
  int slot;
  int rank_lo;
  int rank_hi;
};

struct Scratch {
  std::vector<int> cell_of;
  std::vector<int> rank;
  std::vector<int> counts;
  std::vector<int> start;
  std::vector<int> atoms;
  std::vector<int> scell;
  std::vector<int> scounts;
  std::vector<int> sstart;
  std::vector<int> satoms;
  std::vector<int> lo;
  std::vector<int> hi;
  std::vector<double> dx;
  std::vector<double> dy;
  std::vector<double> dz;
  std::vector<int> lo2;
  std::vector<int> hi2;
  std::vector<double> dx2;
  std::vector<double> dy2;
  std::vector<double> dz2;
  std::vector<Key> key;
  std::vector<std::uint32_t> ord;
  Grid order{};
  bool ordered = false;
};

void fillPairs(const double *positions, const double *box, std::size_t n,
               double cutoff, double search, Scratch &s);

void sortLegacy(Scratch &s);

} // namespace rgpot::nlist::detail
