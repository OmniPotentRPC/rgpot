// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <vector>

#include "rgpot.h"
#include "rgpot/metatomic.h"

using Catch::Matchers::WithinAbs;

// 13 hydrogen atoms from eOn lj38 test (pos.con), positions in Angstrom
static const double lj13_pos[] = {
    50.594227, 52.017165, 52.277482, 51.578540, 52.642148, 51.195222,
    51.243758, 52.919086, 52.218383, 50.490127, 52.736378, 51.417663,
    51.522643, 50.884535, 51.656125, 50.927263, 51.743087, 51.272643,
    51.240676, 50.960436, 50.573042, 51.275727, 52.061877, 50.284160,
    50.434110, 50.976423, 51.879062, 51.695007, 51.919472, 52.038680,
    51.363596, 52.199101, 53.064923, 52.017483, 51.639074, 51.008389,
    51.187843, 51.165217, 52.678225,
};
static const int lj13_atmnrs[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static constexpr int N_ATOMS = 13;

TEST_CASE("Metatomic C API creates a local potential handle", "[metatomic]") {
  rgpot_metatomic_config_t cfg = {};
  cfg.model_path = "data/lj38/lennard-jones.pt";
  cfg.device = "cpu";
  cfg.length_unit = "angstrom";

  auto *pot = rgpot_metatomic_potential_new(&cfg);
  REQUIRE(pot != nullptr);

  std::vector<double> pos(std::begin(lj13_pos), std::end(lj13_pos));
  std::vector<int> atmnrs(std::begin(lj13_atmnrs), std::end(lj13_atmnrs));
  std::array<double, 9> box = {101.9424, 0.0,     0.0,      0.0, 103.1426,
                               0.0,     0.0,      0.0,      102.6055};

  auto input = rgpot_force_input_create(N_ATOMS, pos.data(), atmnrs.data(),
                                        box.data());
  auto output = rgpot_force_out_create();

  auto status = rgpot_potential_calculate(pot, &input, &output);
  REQUIRE(status == RGPOT_SUCCESS);
  REQUIRE_THAT(output.energy, WithinAbs(98374.87753058573, 1e-2));
  REQUIRE(output.forces != nullptr);

  rgpot_tensor_free(output.forces);
  rgpot_force_input_free(&input);
  rgpot_potential_free(pot);
}
