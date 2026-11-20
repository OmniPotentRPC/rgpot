// MIT License
// Copyright 2023--present rgpot developers
//
// Unit expression parser tests. The parser is derived from metatomic-torch
// (metatensor/metatomic PR #173).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rgpot/units.hpp"

using Catch::Matchers::WithinRel;

namespace {
std::uint64_t bits(double value) {
  std::uint64_t out = 0;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}
} // namespace

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
    double factor =
        rgpot::units::unit_conversion_factor("eV/angstrom^3", "hartree/bohr^3");
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
  REQUIRE_THROWS_AS(rgpot::units::unit_conversion_factor("angstrom", "eV"),
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
  double parsed_bohr = rgpot::units::unit_conversion_factor("bohr", "angstrom");
  REQUIRE_THAT(parsed_bohr, WithinRel(rgpot::units::BOHR_TO_ANGSTROM, 1e-11));

  double parsed_hartree = rgpot::units::unit_conversion_factor("hartree", "eV");
  REQUIRE_THAT(parsed_hartree, WithinRel(rgpot::units::HARTREE_TO_EV, 1e-11));

  double parsed_force =
      rgpot::units::unit_conversion_factor("hartree/bohr", "eV/angstrom");
  REQUIRE_THAT(parsed_force,
               WithinRel(rgpot::units::HARTREE_BOHR_TO_EV_ANGSTROM, 1e-11));

  double parsed_dalton =
      rgpot::units::unit_conversion_factor("dalton", "electron_mass");
  REQUIRE_THAT(parsed_dalton,
               WithinRel(rgpot::units::DALTON_IN_ELECTRON_MASSES, 1e-11));

  double parsed_au_time =
      rgpot::units::unit_conversion_factor("hbar", "hartree*s");
  REQUIRE_THAT(parsed_au_time,
               WithinRel(rgpot::units::ATOMIC_UNIT_OF_TIME_S, 1e-11));
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

TEST_CASE("refactored SI factors keep their exact bits", "[units]") {
  const double bohr_m = rgpot::units::unit_conversion_factor("bohr", "meter");
  const double ev_j = rgpot::units::unit_conversion_factor("eV", "joule");
  const double mev_j = rgpot::units::unit_conversion_factor("meV", "joule");
  const double hartree_j = rgpot::units::unit_conversion_factor("hartree", "joule");
  const double dalton_kg = rgpot::units::unit_conversion_factor("dalton", "kg");
  const double emass_kg =
      rgpot::units::unit_conversion_factor("electron_mass", "kg");
  const double charge_c = rgpot::units::unit_conversion_factor("e", "coulomb");
  const double hbar_js =
      rgpot::units::unit_conversion_factor("hbar", "joule*second");

  REQUIRE(bits(bohr_m) == bits(5.29177210903e-11));
  REQUIRE(bits(ev_j) == bits(1.602176634e-19));
  REQUIRE(bits(mev_j) == bits(1.602176634e-19 * 1e-3));
  REQUIRE(bits(hartree_j) == bits(4.3597447222071e-18));
  REQUIRE(bits(dalton_kg) == bits(1.66053906660e-27));
  REQUIRE(bits(emass_kg) == bits(9.1093837015e-31));
  REQUIRE(bits(charge_c) == bits(1.602176634e-19));
  REQUIRE(bits(hbar_js) == bits(1.0545718176461565e-34));
}
