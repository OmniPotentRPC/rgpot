// MIT License
// Copyright 2023--present rgpot developers
// clang-format off
#include <fmt/ostream.h>
#include <cstdlib>
// clang-format on
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/types/adapters/eigen.hpp"
using rgpot::types::AtomMatrix;
using rgpot::types::adapt::eigen::convertToAtomMatrix;
using rgpot::types::adapt::eigen::convertToEigen3d;
using rgpot::types::adapt::eigen::convertToVector;

int main(void) {
  auto ljpot = rgpot::LJPot();
  Eigen::MatrixXd positions(3, 3);
  positions << 1, 2, 3, 1.5, 2.5, 3.5, 4, 5, 6;
  Eigen::VectorXi atomTypes(3);
  atomTypes << 0, 0, 0;
  Eigen::Matrix3d boxMatrix;
  boxMatrix << 15, 0, 0, 0, 20, 0, 0, 0, 30;
  auto [energy, forces] =
      ljpot(convertToAtomMatrix(positions), convertToVector<int>(atomTypes),
            convertToEigen3d(boxMatrix));
  fmt::print("Got energy {}\n Forces:\n{}", energy, fmt::streamed(forces));
  return EXIT_SUCCESS;
}
