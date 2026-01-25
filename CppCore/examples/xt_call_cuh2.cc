// MIT License
// Copyright 2023--present rgpot developers
// clang-format off
#include <fmt/ostream.h>
#include <cstdlib>
// clang-format on
#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/adapters/xtensor.hpp"
using rgpot::types::AtomMatrix;
using rgpot::types::adapt::xtensor::convertToArray3x3;
using rgpot::types::adapt::xtensor::convertToAtomMatrix;
using rgpot::types::adapt::xtensor::convertToVector;

int main(void) {
  auto cuh2pot = rgpot::CuH2Pot();

  xt::xtensor<double, 2> positions{
      {0.63940268750835, 0.90484742551374, 6.97516498544584}, // Cu
      {3.19652040936288, 0.90417430354811, 6.97547796369474}, // Cu
      {8.98363230369760, 9.94703496017833, 7.83556854923689}, // H
      {7.64080177576300, 9.94703114803832, 7.83556986121272}, // H
  };

  xt::xtensor<int, 1> atomTypes{{29, 29, 1, 1}};

  xt::xtensor<double, 2> boxMatrix{
      {15, 0, 0},
      {0, 20, 0},
      {0, 0, 30},
  };

  auto [energy, forces] =
      cuh2pot(convertToAtomMatrix(positions), convertToVector(atomTypes),
              convertToArray3x3(boxMatrix));

  fmt::print("Got energy {}\n Forces:\n{}", energy, fmt::streamed(forces));
  return EXIT_SUCCESS;
}
