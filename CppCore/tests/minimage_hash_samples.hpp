#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/engine/Sha256.hpp"
#include "rgpot/nlist/MinimumImage.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace rgpot::nlist::hashcheck {

// Triclinic cell from the fold micro-benchmark. Displacements include
// +0, -0, a subnormal, a one-ulp shift of 3, and a half-vector tie.
inline std::string folded_displacement_hash() {
  const double cell[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const MinimumImage image(cell, periodic);
  const double tie = std::nextafter(3.0, 4.0);
  const double sub = std::numeric_limits<double>::denorm_min();
  const double samples[][3] = {
      {0.0, 0.0, 0.0},
      {-0.0, 0.0, -0.0},
      {sub, -sub, 0.0},
      {tie, 1.0, -1.0},
      {2.0, 0.1, 0.05},
      {0.2, -0.3, 0.4},
      {1.5, 2.5, -1.0},
      {-20.0, 7.5, 15.0},
      {12.0, -9.0, 4.0},
      {4.0, 0.2, 0.1},
  };
  rgpot::engine::Sha256 hash;
  for (const auto &sample : samples) {
    double x = sample[0], y = sample[1], z = sample[2];
    image.fold(x, y, z);
    hash.update(&x, sizeof x);
    hash.update(&y, sizeof y);
    hash.update(&z, sizeof z);
  }
  return hash.hex_digest();
}

} // namespace rgpot::nlist::hashcheck
