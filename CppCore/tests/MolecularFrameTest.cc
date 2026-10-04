// SPDX-License-Identifier: MIT
#include "rgpot/UmaPot/MolecularFrame.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
  int failures = 0;
  const auto check = [&](bool condition, const char *message) {
    if (!condition) {
      std::cerr << message << '\n';
      ++failures;
    }
  };
  const auto gradient = [&](double displacement, double length) {
    std::array<double, 6> positions{
        12.0 + displacement, 0.0, 0.0, 13.0, 0.0, 0.0};
    const auto cell = rgpot::uma::molecularFrame(positions, length);
    check(cell ==
              std::array<double, 9>{length, 0, 0, 0, length, 0, 0, 0, length},
          "molecular frame changes the specified cell");
    // E = (r - 1)^2 / 2, evaluated from binary32 Cartesian inputs.
    const float bond =
        static_cast<float>(positions[3]) - static_cast<float>(positions[0]);
    return static_cast<double>(1.0f - bond);
  };
  const double h = std::ldexp(1.0, -20);
  for (double length : {25.0, 30.0, 64.0}) {
    const double curvature =
        (gradient(h, length) - gradient(-h, length)) / (2 * h);
    check(curvature == 1.0, "synthetic cell erases a resolved bond curvature");
  }

  std::array<double, 6> first{12.0, 3.0, -4.0, 13.0, 3.5, -4.5};
  auto second = first;
  for (std::size_t i = 0; i < second.size(); i += 3) {
    second[i] += 32.0;
    second[i + 1] -= 16.0;
    second[i + 2] += 8.0;
  }
  rgpot::uma::molecularFrame(first, 25.0);
  rgpot::uma::molecularFrame(second, 25.0);
  check(first == second, "molecular input depends on a uniform translation");
  check(first[3] - first[0] == 1.0 && first[4] - first[1] == 0.5 &&
            first[5] - first[2] == -0.5,
        "molecular frame changes relative Cartesian vectors");
  if (failures == 0)
    std::cout << "molecular frame checks passed\n";
  return failures == 0 ? 0 : 1;
}
