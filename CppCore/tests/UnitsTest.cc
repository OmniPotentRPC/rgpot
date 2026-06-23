// MIT License
// Copyright 2023--present rgpot developers
//
// Unit expression parser tests. The parser is derived from metatomic-torch
// (metatensor/metatomic PR #173).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>

#include "rgpot/units.hpp"

using Catch::Matchers::WithinRel;

TEST_CASE("unit_conversion_factor basic conversions", "[units]") {
  SECTION("identity") {
    REQUIRE(rgpot::units::unit_conversion_factor("angstrom", "angstrom") ==
            1.0);
    REQUIRE(rgpot::units::unit_conversion_factor("eV", "eV") == 1.0);
  }

  SECTION("length") {
    double bohr_to_angstrom =
        rgpot::units::unit_conversion_factor("bohr", "angstrom");
    REQUIRE_THAT(bohr_to_angstrom, WithinRel(0.529177, 1e-4));

    double angstrom_to_nm =
        rgpot::units::unit_conversion_factor("angstrom", "nm");
    REQUIRE_THAT(angstrom_to_nm, WithinRel(0.1, 1e-10));
  }

  SECTION("energy") {
    double hartree_to_ev =
        rgpot::units::unit_conversion_factor("hartree", "eV");
    REQUIRE_THAT(hartree_to_ev, WithinRel(27.2114, 1e-4));

    double kcal_mol_to_ev =
        rgpot::units::unit_conversion_factor("kcal/mol", "eV");
    REQUIRE_THAT(kcal_mol_to_ev, WithinRel(0.04336, 1e-3));
  }

  SECTION("force") {
    double hartree_bohr_to_ev_a =
        rgpot::units::unit_conversion_factor("hartree/bohr", "eV/angstrom");
    REQUIRE_THAT(hartree_bohr_to_ev_a, WithinRel(51.4221, 1e-3));
  }
}

TEST_CASE("unit_conversion_factor compound expressions", "[units]") {
  SECTION("kJ/mol to eV") {
    double factor = rgpot::units::unit_conversion_factor("kJ/mol", "eV");
    REQUIRE_THAT(factor, WithinRel(0.01036, 1e-3));
  }

  SECTION("pressure: eV/A^3 to hartree/bohr^3") {
    double factor = rgpot::units::unit_conversion_factor("eV/angstrom^3",
                                                         "hartree/bohr^3");
    REQUIRE_THAT(factor, WithinRel(0.00544, 1e-2));
  }

  SECTION("whitespace and case are ignored") {
    double a = rgpot::units::unit_conversion_factor("Angstrom", "BOHR");
    double b = rgpot::units::unit_conversion_factor(" angstrom ", " bohr ");
    REQUIRE_THAT(a, WithinRel(b, 1e-12));
  }

  SECTION("power and grouping") {
    // (eV/A) has force dimensions; reciprocal should invert the factor
    double forward =
        rgpot::units::unit_conversion_factor("eV/angstrom", "hartree/bohr");
    double inverse =
        rgpot::units::unit_conversion_factor("hartree/bohr", "eV/angstrom");
    REQUIRE_THAT(forward * inverse, WithinRel(1.0, 1e-10));
  }
}

TEST_CASE("unit_conversion_factor dimension mismatch", "[units]") {
  REQUIRE_THROWS_AS(
      rgpot::units::unit_conversion_factor("angstrom", "eV"),
      std::invalid_argument);
}

TEST_CASE("unit_conversion_factor unknown unit", "[units]") {
  REQUIRE_THROWS_AS(
      rgpot::units::unit_conversion_factor("angstrom", "furlongs"),
      std::invalid_argument);
}

TEST_CASE("validate_unit", "[units]") {
  REQUIRE_NOTHROW(rgpot::units::validate_unit("energy", "eV"));
  REQUIRE_NOTHROW(rgpot::units::validate_unit("force", "eV/angstrom"));
  REQUIRE_NOTHROW(rgpot::units::validate_unit("length", "bohr"));
  REQUIRE_NOTHROW(rgpot::units::validate_unit("pressure", "eV/angstrom^3"));

  REQUIRE_THROWS_AS(rgpot::units::validate_unit("energy", "angstrom"),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::units::validate_unit("force", "eV"),
                    std::invalid_argument);
}

TEST_CASE("constexpr constants match parser", "[units]") {
  double parsed_bohr =
      rgpot::units::unit_conversion_factor("bohr", "angstrom");
  REQUIRE_THAT(parsed_bohr, WithinRel(rgpot::units::BOHR_TO_ANGSTROM, 1e-8));

  double parsed_hartree =
      rgpot::units::unit_conversion_factor("hartree", "eV");
  REQUIRE_THAT(parsed_hartree, WithinRel(rgpot::units::HARTREE_TO_EV, 1e-8));

  double parsed_force = rgpot::units::unit_conversion_factor("hartree/bohr",
                                                             "eV/angstrom");
  REQUIRE_THAT(parsed_force,
               WithinRel(rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM, 1e-8));
}

TEST_CASE("round-trip conversions are inverses", "[units]") {
  const char *pairs[][2] = {
      {"angstrom", "bohr"},
      {"eV", "hartree"},
      {"eV/angstrom", "hartree/bohr"},
      {"kcal/mol", "eV"},
  };
  for (const auto &pair : pairs) {
    double f = rgpot::units::unit_conversion_factor(pair[0], pair[1]);
    double g = rgpot::units::unit_conversion_factor(pair[1], pair[0]);
    REQUIRE_THAT(f * g, WithinRel(1.0, 1e-10));
  }
}
