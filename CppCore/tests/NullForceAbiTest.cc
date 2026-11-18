// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>

#include <string>

#include "rgpot/plugin/PluginRegistry.hpp"

namespace {

enum class Spoil { Positions, AtomicNumbers, Box, Forces, Count };

const char *expectedMessage(Spoil spoil) {
  switch (spoil) {
  case Spoil::Positions:
    return "invalid parameter: positions is null";
  case Spoil::AtomicNumbers:
    return "invalid parameter: atomic numbers are null";
  case Spoil::Box:
    return "invalid parameter: box is null";
  case Spoil::Forces:
    return "invalid parameter: forces are null";
  case Spoil::Count:
    return "invalid parameter: nAtoms must be positive";
  }
  return "";
}

void requireRejected(const char *name, const char *config, Spoil spoil) {
  const auto *plugin = rgpot::plugin::PluginRegistry::instance().find(name);
  REQUIRE(plugin != nullptr);
  const rgpot_plugin_descriptor_t *desc = plugin->desc;
  REQUIRE(desc != nullptr);
  REQUIRE(desc->calculate != nullptr);
  REQUIRE(desc->last_error != nullptr);

  rgpot_plugin_instance *raw = nullptr;
  REQUIRE(desc->create(config, &raw) == RGPOT_PLUGIN_OK);
  REQUIRE(raw != nullptr);
  struct Guard {
    const rgpot_plugin_descriptor_t *desc;
    rgpot_plugin_instance *raw;
    ~Guard() {
      if (raw != nullptr) {
        desc->destroy(raw);
      }
    }
  } guard{desc, raw};

  double positions[6] = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  int atomic[2] = {29, 28};
  double cell[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  double forces[6] = {};
  rgpot_plugin_input_t input{.n_atoms = 2,
                             .positions = positions,
                             .atomic_numbers = atomic,
                             .cell = cell};
  rgpot_plugin_output_t output{.energy = 0.0, .variance = 0.0, .forces = forces};
  if (spoil == Spoil::Positions) {
    input.positions = nullptr;
  } else if (spoil == Spoil::AtomicNumbers) {
    input.atomic_numbers = nullptr;
  } else if (spoil == Spoil::Box) {
    input.cell = nullptr;
  } else if (spoil == Spoil::Forces) {
    output.forces = nullptr;
  } else {
    input.n_atoms = 0;
  }

  REQUIRE(desc->calculate(raw, &input, &output) == RGPOT_PLUGIN_INVALID_PARAM);
  const char *message = desc->last_error(raw);
  REQUIRE(message != nullptr);
  REQUIRE(std::string(message) == expectedMessage(spoil));
}

} // namespace

#define RGPOT_NULL_CASE(name, config, spoil, label)                            \
  TEST_CASE(name " rejects " label, "[null-force]") {                         \
    requireRejected(name, config, spoil);                                      \
  }

RGPOT_NULL_CASE("lennard-jones", nullptr, Spoil::Positions, "null positions")
RGPOT_NULL_CASE("lennard-jones", nullptr, Spoil::AtomicNumbers,
                "null atomic numbers")
RGPOT_NULL_CASE("lennard-jones", nullptr, Spoil::Box, "null box")
RGPOT_NULL_CASE("lennard-jones", nullptr, Spoil::Forces, "null forces")
RGPOT_NULL_CASE("lennard-jones", nullptr, Spoil::Count, "non-positive nAtoms")

RGPOT_NULL_CASE("classical-morse", nullptr, Spoil::Positions, "null positions")
RGPOT_NULL_CASE("classical-morse", nullptr, Spoil::AtomicNumbers,
                "null atomic numbers")
RGPOT_NULL_CASE("classical-morse", nullptr, Spoil::Box, "null box")
RGPOT_NULL_CASE("classical-morse", nullptr, Spoil::Forces, "null forces")
RGPOT_NULL_CASE("classical-morse", nullptr, Spoil::Count, "non-positive nAtoms")

RGPOT_NULL_CASE("sutton-chen", nullptr, Spoil::Positions, "null positions")
RGPOT_NULL_CASE("sutton-chen", nullptr, Spoil::AtomicNumbers,
                "null atomic numbers")
RGPOT_NULL_CASE("sutton-chen", nullptr, Spoil::Box, "null box")
RGPOT_NULL_CASE("sutton-chen", nullptr, Spoil::Forces, "null forces")
RGPOT_NULL_CASE("sutton-chen", nullptr, Spoil::Count, "non-positive nAtoms")

RGPOT_NULL_CASE("eam-funcfl", "CppCore/tests/data/eam_table/Cu_u3.eam",
                Spoil::Positions, "null positions")
RGPOT_NULL_CASE("eam-funcfl", "CppCore/tests/data/eam_table/Cu_u3.eam",
                Spoil::AtomicNumbers, "null atomic numbers")
RGPOT_NULL_CASE("eam-funcfl", "CppCore/tests/data/eam_table/Cu_u3.eam",
                Spoil::Box, "null box")
RGPOT_NULL_CASE("eam-funcfl", "CppCore/tests/data/eam_table/Cu_u3.eam",
                Spoil::Forces, "null forces")
RGPOT_NULL_CASE("eam-funcfl", "CppCore/tests/data/eam_table/Cu_u3.eam",
                Spoil::Count, "non-positive nAtoms")

RGPOT_NULL_CASE("eam-setfl", "CppCore/tests/data/eam_table/CuNi.eam.alloy",
                Spoil::Positions, "null positions")
RGPOT_NULL_CASE("eam-setfl", "CppCore/tests/data/eam_table/CuNi.eam.alloy",
                Spoil::AtomicNumbers, "null atomic numbers")
RGPOT_NULL_CASE("eam-setfl", "CppCore/tests/data/eam_table/CuNi.eam.alloy",
                Spoil::Box, "null box")
RGPOT_NULL_CASE("eam-setfl", "CppCore/tests/data/eam_table/CuNi.eam.alloy",
                Spoil::Forces, "null forces")
RGPOT_NULL_CASE("eam-setfl", "CppCore/tests/data/eam_table/CuNi.eam.alloy",
                Spoil::Count, "non-positive nAtoms")
