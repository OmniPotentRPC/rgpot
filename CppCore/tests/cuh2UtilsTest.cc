// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>

#include "rgpot/CuH2/cuh2Utils.hpp"

TEST_CASE("perturb_positions correctly perturbs hydrogen atom positions and "
          "sets new z-coordinate",
          "[perturb_positions]") {
  xt::xtensor<double, 2> base_positions = {{0, 0, 0}, {1, 0, 0}, {0.5, 0.5, 2}};
  xt::xtensor<int, 1> atmNumVec = {1, 1, 29}; // Hydrogens and one Copper
  double hcu_dist = 1.5;
  double hh_dist = 1.0;

  auto perturbed_positions = rgpot::cuh2::utils::xts::perturb_positions(
      base_positions, atmNumVec, hcu_dist, hh_dist);

  // Expected behavior
  REQUIRE_THAT(perturbed_positions(0, 2),
               Catch::Matchers::WithinAbs(2.0 + hcu_dist, 1e-4));
  REQUIRE_THAT(perturbed_positions(1, 2),
               Catch::Matchers::WithinAbs(2.0 + hcu_dist, 1e-4));

  // Check if the distance between hydrogens is approximately 1.0
  auto hDist = xt::linalg::norm(xt::row(perturbed_positions, 0) -
                                xt::row(perturbed_positions, 1));
  REQUIRE_THAT(hDist, Catch::Matchers::WithinAbs(hh_dist, 1e-4));
}

TEST_CASE("calculateDistances calculates distances correctly",
          "[calculateDistances]") {
  xt::xtensor<double, 2> positions = {
      {0.0, 0.0, 3.0}, {1.0, 0.0, 3.0}, {0.5, 0.5, 2.0}};
  xt::xtensor<int, 1> atmNumVec = {1, 1, 29}; // Hydrogens and one Copper

  auto distances =
      rgpot::cuh2::utils::xts::calculateDistances(positions, atmNumVec);
  // Distance between hydrogens
  REQUIRE_THAT(distances.first, Catch::Matchers::WithinAbs(1.0, 1e-4));
  // Distance from H to top Cu layer
  REQUIRE_THAT(distances.second, Catch::Matchers::WithinAbs(1.0, 1e-4));
}

// Additional tests to cover error handling and unexpected inputs
TEST_CASE("perturb_positions throws when unexpected atom numbers arepresent",
          "[perturb_positions]") {
  xt::xtensor<double, 2> base_positions = {{0, 0, 0}, {1, 0, 0}};
  xt::xtensor<int, 1> atmNumVec = {1, 30}; // Invalid atom number

  REQUIRE_THROWS_AS(rgpot::cuh2::utils::xts::perturb_positions(
                        base_positions, atmNumVec, 1.5, 1.0),
                    std::runtime_error);
}

TEST_CASE(
    "calculateDistances throws when not exactly two hydrogens are present",
    "[calculateDistances]") {
  xt::xtensor<double, 2> positions = {{0.0, 0.0, 3.0}, {0.5, 0.5, 2.0}};
  xt::xtensor<int, 1> atmNumVec = {1, 29}; // Not two Hydrogens

  REQUIRE_THROWS_AS(
      rgpot::cuh2::utils::xts::calculateDistances(positions, atmNumVec),
      std::runtime_error);
}
