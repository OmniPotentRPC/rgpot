#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file HalfShell.hpp
 * @brief Skin record for a three-body call.
 *
 * The pairs come from the shared neighbour layer. This record only
 * remembers the geometry a positive skin was built at, so a later call
 * can tell a reused binning from a fresh one.
 */

#include <cstddef>
#include <vector>

namespace rgpot::nlist {

struct NeighborList {
  double skin = 0;
  void clear() { built = false; }

  std::vector<double> ref;
  double box[9]{};
  std::size_t n = 0;
  double cutoff = 0;
  double skin_used = 0;
  bool built = false;
};

} // namespace rgpot::nlist
