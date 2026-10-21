// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Equivalence pins for the Fortran 2018 potential kernels.
 *
 * The geometries and reference energies come from eOn's suites
 * (client/unit_tests/{SiPotTest,EAMAlTest,FeHeTest}.cpp with the
 * fixtures under client/unit_tests/data/systems), carried over at the
 * tolerances those suites used. They pin the ports against the legacy
 * kernels: the rearrangement into gather form and the parameter/derived
 * type restructuring must not move any number here.
 *
 * Physics-level checks (translation invariance, vanishing net force,
 * analytic forces against central differences) live in the Fortran test
 * programs beside the kernels, where a failure points at the kernel
 * rather than the bindings.
 */

#include <catch2/catch_all.hpp>
#include <cmath>
#include <vector>

#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

/// eOn fixture data/systems/si_diamond/pos.con: eight silicon atoms in a
/// 20 Angstrom cube.
AtomMatrix siDiamond() {
  return AtomMatrix{{7.28500, 7.28500, 7.28500},   {8.64250, 8.64250, 8.64250},
                    {10.00000, 10.00000, 7.28500}, {11.35750, 11.35750, 8.64250},
                    {10.00000, 7.28500, 10.00000}, {11.35750, 8.64250, 11.35750},
                    {7.28500, 10.00000, 10.00000}, {8.64250, 11.35750, 11.35750}};
}

/// eOn fixture data/systems/al_fcc/pos.con: four aluminium atoms.
AtomMatrix alFcc() {
  return AtomMatrix{{7.97500, 7.97500, 7.97500},
                    {10.00000, 10.00000, 7.97500},
                    {10.00000, 7.97500, 10.00000},
                    {7.97500, 10.00000, 10.00000}};
}

/// eOn fixture data/systems/fe_bcc/pos.con: sixteen iron atoms.
AtomMatrix feBcc() {
  return AtomMatrix{
      {7.13000, 7.13000, 7.13000},    {8.56500, 8.56500, 8.56500},
      {7.13000, 7.13000, 10.00000},   {8.56500, 8.56500, 11.43500},
      {7.13000, 10.00000, 7.13000},   {8.56500, 11.43500, 8.56500},
      {7.13000, 10.00000, 10.00000},  {8.56500, 11.43500, 11.43500},
      {10.00000, 7.13000, 7.13000},   {11.43500, 8.56500, 8.56500},
      {10.00000, 7.13000, 10.00000},  {11.43500, 8.56500, 11.43500},
      {10.00000, 10.00000, 7.13000},  {11.43500, 11.43500, 8.56500},
      {10.00000, 10.00000, 10.00000}, {11.43500, 11.43500, 11.43500}};
}

/// The 20 Angstrom cube every fixture above sits in.
std::array<std::array<double, 3>, 3> cube20() {
  return {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
}

double maxForceNorm(const AtomMatrix &forces) {
  double worst = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

void requireNoNetForce(const AtomMatrix &forces) {
  double sums[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t d = 0; d < 3; ++d) {
      sums[d] += forces(i, d);
    }
  }
  for (const double s : sums) {
    REQUIRE(std::abs(s) < 1e-6);
  }
}

} // namespace

TEST_CASE("SW silicon matches the eOn reference", "[fortran][sw]") {
  rgpot::fortranpots::SWPot pot;
  const std::vector<int> types(8, 14);
  auto [energy, forces, variance] = pot(siDiamond(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-16.204955, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.005232, 1e-2));
  requireNoNetForce(forces);
}

TEST_CASE("Tersoff silicon matches the eOn reference", "[fortran][tersoff]") {
  rgpot::fortranpots::TersoffPot pot;
  const std::vector<int> types(8, 14);
  auto [energy, forces, variance] = pot(siDiamond(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-17.440266, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(1.082719, 1e-3));
  requireNoNetForce(forces);
}

TEST_CASE("EDIP silicon matches the eOn reference", "[fortran][edip]") {
  rgpot::fortranpots::EDIPPot pot;
  const std::vector<int> types(8, 14);
  auto [energy, forces, variance] = pot(siDiamond(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-18.838135, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.730971, 1e-3));
  requireNoNetForce(forces);
}

TEST_CASE("Lenosky silicon matches the eOn reference", "[fortran][lenosky]") {
  rgpot::fortranpots::LenoskyPot pot;
  const std::vector<int> types(8, 14);
  auto [energy, forces, variance] = pot(siDiamond(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-17.284558, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.456933, 1e-3));
  requireNoNetForce(forces);
}

TEST_CASE("EAM aluminium matches the eOn reference", "[fortran][eamal]") {
  rgpot::fortranpots::EAMAlPot pot;
  const std::vector<int> types(4, 13);
  auto [energy, forces, variance] = pot(alFcc(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-5.217864, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.968647, 1e-3));
  requireNoNetForce(forces);
}

TEST_CASE("FeHe iron matches the eOn reference", "[fortran][fehe]") {
  rgpot::fortranpots::FeHePot pot;
  const std::vector<int> types(16, 26);
  auto [energy, forces, variance] = pot(feBcc(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-43.959774, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(1.245094, 1e-3));
  requireNoNetForce(forces);
}

TEST_CASE("Fortran potentials report per-instance reentrancy",
          "[fortran][caps]") {
  // Each instance owns its neighbour table, so separate instances evaluate
  // concurrently and multi-image callers keep one per image.
  for (const rgpot::PotCaps caps :
       {rgpot::fortranpots::SWPot{}.caps(), rgpot::fortranpots::EAMAlPot{}.caps(),
        rgpot::fortranpots::FeHePot{}.caps(),
        rgpot::fortranpots::CuH2Pot{}.caps()}) {
    REQUIRE(caps.reentrancy == rgpot::Reentrancy::PerInstance);
    REQUIRE(caps.perImageInstances);
  }
}
