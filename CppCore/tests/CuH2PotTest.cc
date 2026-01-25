// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>

#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

TEST_CASE("CuH2Pot Energy and Forces", "[CuH2Pot]") {
  using rgpot::types::AtomMatrix;
  auto cuh2pot = rgpot::CuH2Pot();
  AtomMatrix positions{
      {0.63940268750835, 0.90484742551374, 6.97516498544584}, // Cu
      {3.19652040936288, 0.90417430354811, 6.97547796369474}, // Cu
      {8.98363230369760, 9.94703496017833, 7.83556854923689}, // H
      {7.64080177576300, 9.94703114803832, 7.83556986121272}  // H
  };
  std::vector<int> atmtypes{29, 29, 1, 1};
  std::array<std::array<double, 3>, 3> box{{
      {15.345599999999999, 0, 0}, //
      {0, 21.702000000000002, 0}, //
      {0, 0, 100.00000000000000}  //
  }};
  auto [energy, forces] = cuh2pot(positions, atmtypes, box);

  double expected_energy = -2.7114096242662238;
  AtomMatrix expected_forces{
      {1.4919411183978113, -3.9273058476626193E-004, 1.8260603127768336E-004},
      {-1.4919411183978113, 3.9273058476626193E-004, -1.8260603127768336E-004},
      {-4.9118653085630006, -1.3944215503304855E-005, 4.7990036210569753E-006},
      {4.9118653085630006, 1.3944215503304855E-005, -4.7990036210569753E-006}};

  REQUIRE_THAT(energy, Catch::Matchers::WithinAbs(expected_energy, 1e-6));

  // For forces, you may need to loop over each element if there isn't a direct
  // Catch2 matcher for matrices.
  for (int i = 0; i < forces.rows(); ++i) {
    for (int j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j),
                   Catch::Matchers::WithinAbs(expected_forces(i, j), 1e-6));
    }
  }
}
