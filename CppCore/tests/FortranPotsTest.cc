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
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <thread>
#include <vector>

#include "rgpot/ForceStructs.hpp"
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

TEST_CASE("EAM aluminium matches the EON r2549 Fortran energy",
          "[fortran][eamal]") {
  rgpot::fortranpots::EAMAlPot pot;
  const std::vector<int> types(4, 13);
  // eOn client/unit_tests/data/systems/al_fcc/pos.con: four atoms in a
  // 20 Angstrom cube. EON r2549 `eonclient` point job, potential eam_al,
  // prints UTOT -5.217864006173226.
  auto [energy, forces, variance] = pot(alFcc(), types, cube20());
  (void)variance;

  REQUIRE_THAT(energy, WithinRel(-5.217864, 1e-4));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.968647, 1e-3));
  // The relative pin above still passes if the parameters keep the extra
  // printed digits. Those digits lower this energy by 1.76e-6 eV.
  REQUIRE(energy == Catch::Approx(-5.217864006173226).margin(1e-12));
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

namespace {

/// 4 x 4 x 4 fcc aluminium cells (256 atoms) with a deterministic wobble,
/// periodic in a cube of 4 lattice constants.
std::vector<double> alBlock(double phase) {
  const double a = 4.05;
  static const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  std::vector<double> R;
  for (int ix = 0; ix < 4; ++ix)
    for (int iy = 0; iy < 4; ++iy)
      for (int iz = 0; iz < 4; ++iz)
        for (const auto &b : basis) {
          const auto k = static_cast<double>(R.size());
          R.push_back((ix + b[0]) * a + 0.08 * std::sin(1.3 * k + phase));
          R.push_back((iy + b[1]) * a + 0.08 * std::sin(1.7 * k + phase));
          R.push_back((iz + b[2]) * a + 0.08 * std::sin(2.9 * k + phase));
        }
  return R;
}

struct Result {
  double energy = 0.0;
  std::vector<double> F;
};

Result evalEam(const rgpot::fortranpots::EAMAlPot &pot,
               const std::vector<double> &R) {
  static const double box[9] = {16.2, 0, 0, 0, 16.2, 0, 0, 0, 16.2};
  const std::vector<int> types(R.size() / 3, 13);
  Result out;
  out.F.assign(R.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = R.size() / 3,
                       .pos = R.data(),
                       .atmnrs = types.data(),
                       .box = box};
  rgpot::ForceOut fo{.F = out.F.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(in, &fo);
  out.energy = fo.energy;
  return out;
}

} // namespace

TEST_CASE("EAM aluminium instances evaluate concurrently", "[fortran][eamal]") {
  // Serial reference for two geometry families.
  const auto ra = alBlock(0.0);
  const auto rb = alBlock(1.0);
  rgpot::fortranpots::EAMAlPot ref;
  const Result wantA = evalEam(ref, ra);
  const Result wantB = evalEam(ref, rb);

  // Two instances, one per family, on two threads at once, many times;
  // each table follows its own family. Results agree to the bit with the
  // serial ones: rows are ordered by partner whatever the table's history.
  rgpot::fortranpots::EAMAlPot potA;
  rgpot::fortranpots::EAMAlPot potB;
  std::atomic<int> mismatches{0};
  auto worker = [&](const rgpot::fortranpots::EAMAlPot &pot,
                    const std::vector<double> &R, const Result &want) {
    for (int k = 0; k < 40; ++k) {
      const Result got = evalEam(pot, R);
      if (got.energy != want.energy || got.F != want.F) {
        ++mismatches;
      }
    }
  };
  std::thread ta(worker, std::cref(potA), std::cref(ra), std::cref(wantA));
  std::thread tb(worker, std::cref(potB), std::cref(rb), std::cref(wantB));
  ta.join();
  tb.join();
  REQUIRE(mismatches.load() == 0);
}

TEST_CASE("EAM aluminium forces equal minus the energy gradient",
          "[fortran][eamal]") {
  rgpot::fortranpots::EAMAlPot pot;
  auto R = alBlock(0.4);
  const Result base = evalEam(pot, R);
  const double h = 1e-5;
  // Every coordinate of the first eight atoms, against central differences.
  for (std::size_t k = 0; k < 24; ++k) {
    const double x0 = R[k];
    R[k] = x0 + h;
    const double ep = evalEam(pot, R).energy;
    R[k] = x0 - h;
    const double em = evalEam(pot, R).energy;
    R[k] = x0;
    const double fd = -(ep - em) / (2.0 * h);
    REQUIRE_THAT(base.F[k], Catch::Matchers::WithinAbs(
                                fd, 1e-6 * std::max(1.0, std::abs(fd))));
  }
}
