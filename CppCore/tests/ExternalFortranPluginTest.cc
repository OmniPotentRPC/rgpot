// MIT License
// Copyright 2023--present rgpot developers
//
// Integration tests for external plugin .so files.
// Each test loads a plugin via PluginRegistry::load_plugin(), creates
// an instance via the bridge, and verifies energy/forces.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Helper: get plugin .so path from RGPOT_TEST_PLUGIN_DIR env var
static std::string plugin_path(const char *filename) {
  const char *dir = std::getenv("RGPOT_TEST_PLUGIN_DIR");
  if (!dir)
    dir = ".";
  std::string library(filename);
#ifdef _WIN32
  library.replace(library.size() - 3, 3, ".dll");
#elif defined(__APPLE__)
  library.replace(library.size() - 3, 3, ".dylib");
#endif
  return std::string(dir) + "/" + library;
}

// Two-atom test system
static const double two_pos[] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
static const int two_atm[] = {1, 1};
static const std::array<std::array<double, 3>, 3> big_box = {
    {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};

static AtomMatrix make_pos(const double *data, int n) {
  AtomMatrix m(n, 3);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < 3; ++j)
      m(i, j) = data[i * 3 + j];
  return m;
}

// ---- Coulomb plugin (Fortran) ----

TEST_CASE("External Coulomb plugin: two protons", "[plugin][external]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  reg.load_plugin(plugin_path("librgpot_coulomb.so"));
  const auto *p = reg.find("coulomb");
  REQUIRE(p != nullptr);

  auto pot = rgpot::plugin::create_from_plugin(p->desc);
  auto positions = make_pos(two_pos, 2);
  std::vector<int> types(two_atm, two_atm + 2);

  auto [energy, forces, unused_variance] = (*pot)(positions, types, big_box);

  // E = ke * q1 * q2 / r = 14.3996 * 1 * 1 / 1.5
  double expected_e = 14.3996448915 / 1.5;
  REQUIRE_THAT(energy, WithinRel(expected_e, 1e-6));

  // Repulsive: force on atom 0 should be negative (pushed away from atom 1)
  REQUIRE(forces(0, 0) < 0.0);
  // Newton's third law
  REQUIRE_THAT(forces(0, 0) + forces(1, 0), WithinAbs(0.0, 1e-10));
}
