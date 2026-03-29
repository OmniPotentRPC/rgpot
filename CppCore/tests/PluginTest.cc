// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <vector>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/plugin/PluginBridge.hpp"
#include "rgpot/plugin/PluginRegistry.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Two-atom LJ system for comparison
static const double lj2_pos[] = {0.0, 0.0, 0.0, 1.5, 0.0, 0.0};
static const int lj2_atmnrs[] = {1, 1};

TEST_CASE("Plugin LJ matches direct LJ", "[plugin]") {
  // Direct LJ calculation
  rgpot::LJPot direct;
  AtomMatrix positions(2, 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      positions(i, j) = lj2_pos[i * 3 + j];
  std::vector<int> atmtypes(lj2_atmnrs, lj2_atmnrs + 2);
  std::array<std::array<double, 3>, 3> box = {
      {{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
  auto [direct_energy, direct_forces] = direct(positions, atmtypes, box);

  // Plugin LJ (via builtin shim)
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  // The LJ builtin should be registered
  const auto *lj = reg.find("lennard-jones");
  REQUIRE(lj != nullptr);
  REQUIRE(lj->desc->api_version == RGPOT_PLUGIN_API_VERSION);

  auto plugin_pot =
      rgpot::plugin::create_from_plugin(lj->desc, "{}");
  auto [plugin_energy, plugin_forces] =
      (*plugin_pot)(positions, atmtypes, box);

  REQUIRE_THAT(plugin_energy, WithinAbs(direct_energy, 1e-12));
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      REQUIRE_THAT(plugin_forces(i, j),
                   WithinAbs(direct_forces(i, j), 1e-12));
    }
  }
}

TEST_CASE("Plugin registry lists builtins", "[plugin]") {
  auto &reg = rgpot::plugin::PluginRegistry::instance();
  auto names = reg.list_plugins();
  bool found_lj = false;
  for (const auto &n : names) {
    if (n == "lennard-jones")
      found_lj = true;
  }
  REQUIRE(found_lj);
}
