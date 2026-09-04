// MIT License
// Copyright 2023--present rgpot developers
// clang-format off
#include <fmt/ostream.h>
#include <cstdlib>
// clang-format on
#include <array>
#include <memory>
#include <vector>

#include "rgpot/ExprPot/ExprPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

int main(void) {
  std::vector<rgpot::ExprPot::Term> terms;
  terms.emplace_back("lj", std::make_unique<rgpot::LJPot>());
  terms.emplace_back("morse", std::make_unique<rgpot::MorsePot>());
  rgpot::ExprPot pot("0.5*lj + morse", std::move(terms));

  // Two-atom slice of the eigen_call_ljpot.cc fixture (same box).
  AtomMatrix positions{
      {1.0, 2.0, 3.0},
      {1.5, 2.5, 3.5},
  };
  std::vector<int> atomTypes{0, 0};
  std::array<std::array<double, 3>, 3> boxMatrix{{
      {15, 0, 0},
      {0, 20, 0},
      {0, 0, 30},
  }};
  auto [energy, forces, variance] = pot(positions, atomTypes, boxMatrix);
  (void)variance;
  fmt::print("Got energy {}\n Forces:\n{}", energy, fmt::streamed(forces));
  return EXIT_SUCCESS;
}
