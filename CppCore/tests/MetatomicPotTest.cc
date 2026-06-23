// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <vector>

#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// 13 hydrogen atoms from eOn lj38 test (pos.con), positions in Angstrom
// Cell: 101.9424 x 103.1426 x 102.6055 (orthogonal, 90 deg)
// clang-format off
static const double lj13_pos[] = {
    50.594227, 52.017165, 52.277482,
    51.578540, 52.642148, 51.195222,
    51.243758, 52.919086, 52.218383,
    50.490127, 52.736378, 51.417663,
    51.522643, 50.884535, 51.656125,
    50.927263, 51.743087, 51.272643,
    51.240676, 50.960436, 50.573042,
    51.275727, 52.061877, 50.284160,
    50.434110, 50.976423, 51.879062,
    51.695007, 51.919472, 52.038680,
    51.363596, 52.199101, 53.064923,
    52.017483, 51.639074, 51.008389,
    51.187843, 51.165217, 52.678225,
};
// clang-format on
static const int lj13_atmnrs[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static constexpr int N_ATOMS = 13;

TEST_CASE("MetatomicPot LJ energy and forces", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  auto [energy, forces] = pot(positions, atmtypes, box);

  // Reference values from eOn MetatomicTest (lennard-jones.pt model)
  double expected_energy = 98374.87753058573;
  REQUIRE_THAT(energy, WithinAbs(expected_energy, 1e-2));

  // clang-format off
  double expected_forces_flat[] = {
      -90017.67874663,  14048.63238980,  42667.56998833,
       51715.62564964,  81020.09231365, -33288.72879168,
       13893.56176940,  89793.76258628,  33937.11367155,
      -71755.76235155,  52324.79462503, -32402.45955531,
       46268.61994191, -89837.93451292,  11509.42733453,
      -67770.69814970,  -6678.90441423, -33619.79643378,
      -17329.67691782, -68054.08864133, -59929.39232182,
      -14176.97088206,  30833.36510504, -85864.44933633,
      -75239.50566824, -57104.74927750,  -3715.98865599,
       90169.57785920,   5214.90295971,  30786.57650636,
       20281.84422414,  21840.57389606,  85752.46910389,
      104913.11853287, -11120.68664665, -29664.02315515,
        9047.94473885, -62279.76038293,  73831.68164540,
  };
  // clang-format on

  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(forces(i, j),
                   WithinAbs(expected_forces_flat[i * 3 + j], 1e-2));
    }
  }
}

TEST_CASE("MetatomicPot forces sum to zero", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  auto [energy, forces] = pot(positions, atmtypes, box);

  // Newton's third law: force sum should be near zero
  double fx_sum = 0.0, fy_sum = 0.0, fz_sum = 0.0;
  for (int i = 0; i < N_ATOMS; ++i) {
    fx_sum += forces(i, 0);
    fy_sum += forces(i, 1);
    fz_sum += forces(i, 2);
  }
  REQUIRE_THAT(fx_sum, WithinAbs(0.0, 1.0));
  REQUIRE_THAT(fy_sum, WithinAbs(0.0, 1.0));
  REQUIRE_THAT(fz_sum, WithinAbs(0.0, 1.0));
}

TEST_CASE("MetatomicPot is deterministic across repeated calls",
          "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";
  rgpot::MetatomicPot pot(cfg);

  AtomMatrix positions(N_ATOMS, 3);
  for (int i = 0; i < N_ATOMS; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj13_pos[i * 3 + j];

  std::vector<int> atmtypes(lj13_atmnrs, lj13_atmnrs + N_ATOMS);
  std::array<std::array<double, 3>, 3> box = {
      {{101.9424, 0.0, 0.0}, {0.0, 103.1426, 0.0}, {0.0, 0.0, 102.6055}}};

  // Second call exercises the cached atomic_types tensor path
  auto [e1, f1] = pot(positions, atmtypes, box);
  auto [e2, f2] = pot(positions, atmtypes, box);

  REQUIRE_THAT(e1, WithinAbs(e2, 1e-8));
  for (int i = 0; i < N_ATOMS; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(f1(i, j), WithinAbs(f2(i, j), 1e-6));
    }
  }
}

TEST_CASE("MetatomicPot missing model path throws", "[metatomic]") {
  rgpot::MetatomicConfig cfg;
  cfg.model_path = "data/lj38/does-not-exist.pt";
  cfg.device = "cpu";
  REQUIRE_THROWS(rgpot::MetatomicPot(cfg));
}
