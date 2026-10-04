// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>

namespace rgpot::uma {

/// Centre molecular inputs at zero before conversion to model precision.
inline std::array<double, 9> molecularFrame(std::span<double> positions,
                                            double length) {
  if (positions.empty() || positions.size() % 3 != 0 || !(length > 0.0) ||
      !std::isfinite(length))
    throw std::invalid_argument(
        "molecular frame requires atoms and a finite cell");
  const std::size_t atoms = positions.size() / 3;
  std::array<double, 3> center{};
  for (std::size_t i = 0; i < atoms; ++i)
    for (std::size_t d = 0; d < 3; ++d)
      center[d] += positions[3 * i + d];
  for (double &coordinate : center)
    coordinate /= static_cast<double>(atoms);
  for (std::size_t i = 0; i < atoms; ++i)
    for (std::size_t d = 0; d < 3; ++d)
      positions[3 * i + d] -= center[d];
  return {length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, length};
}

} // namespace rgpot::uma
