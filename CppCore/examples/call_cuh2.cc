// MIT License
// Copyright 2023--present rgpot developers
// clang-format off
#include <fmt/ostream.h>
#include <cstdlib>
// clang-format on
#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

int main(void) {
  auto cuh2pot = rgpot::CuH2Pot();
  AtomMatrix positions{
      {0.63940268750835, 0.90484742551374, 6.97516498544584}, // Cu
      {3.19652040936288, 0.90417430354811, 6.97547796369474}, // Cu
      {8.98363230369760, 9.94703496017833, 7.83556854923689}, // H
      {7.64080177576300, 9.94703114803832, 7.83556986121272}, // H
  };

  std::vector<int> atomTypes{29, 29, 1, 1};
  std::array<std::array<double, 3>, 3> boxMatrix{{
      //
      {15, 0, 0}, //
      {0, 20, 0}, //
      {0, 0, 30}  //
  }};
  auto [energy, forces] = cuh2pot(positions, atomTypes, boxMatrix);
  fmt::print("Got energy {}\n Forces:\n{}", energy, fmt::streamed(forces));
  return EXIT_SUCCESS;
}
