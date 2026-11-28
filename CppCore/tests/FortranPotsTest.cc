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
#include "rgpot/ForceStructs.hpp"
#include <barrier>
#include <thread>
#include <functional>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "TestFixtures.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using Catch::Matchers::ContainsSubstring;
using rgpot::test_fixtures::maxForceNorm;
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
  return rgpot::test_fixtures::cubicCell(20.0);
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

namespace {

struct SwStress {
  double energy = 0.0;
  std::array<double, 9> stress{};
};

template <typename Pot>
SwStress swStress(const Pot &pot, const std::vector<double> &R,
                  const std::array<double, 9> &box) {
  std::vector<double> F(R.size(), 0.0);
  const std::vector<int> types(R.size() / 3, 14);
  rgpot::ForceInput in{.nAtoms = R.size() / 3,
                       .pos = R.data(),
                       .atmnrs = types.data(),
                       .box = box.data()};
  rgpot::ForceOut fo{.F = F.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(in, &fo);
  REQUIRE(fo.has_stress == 1);
  SwStress out{fo.energy, {}};
  std::copy(fo.stress, fo.stress + 9, out.stress.begin());
  return out;
}

/// 216 diamond silicon atoms at fractions of `box`, moved by up to
/// `wobble` Angstrom along each Cartesian direction.
std::vector<double> siBlock(const std::array<double, 9> &box, double wobble) {
  static const double basis[8][3] = {
      {0.00, 0.00, 0.00}, {0.00, 0.50, 0.50}, {0.50, 0.00, 0.50},
      {0.50, 0.50, 0.00}, {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75},
      {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  std::vector<double> R;
  for (int ix = 0; ix < 3; ++ix)
    for (int iy = 0; iy < 3; ++iy)
      for (int iz = 0; iz < 3; ++iz)
        for (const auto &b : basis) {
          const double f[3] = {(ix + b[0]) / 3, (iy + b[1]) / 3,
                               (iz + b[2]) / 3};
          const auto k = static_cast<double>(R.size() / 3);
          for (int c = 0; c < 3; ++c) {
            R.push_back(f[0] * box[c] + f[1] * box[3 + c] + f[2] * box[6 + c] +
                        wobble * std::sin((1.3 + 0.4 * c) * k + c));
          }
        }
  return R;
}

/// (1/V) dE/d(eps_row,col) by the fourth-order central difference, step
/// 2e-4. The strain maps every position and cell row r to r (I + eps).
double swStrainDifference(const std::vector<double> &R,
                          const std::array<double, 9> &box, int row, int col) {
  constexpr double h = 2e-4;
  const rgpot::fortranpots::SWPot pot;
  auto energyAt = [&](double eps) {
    std::vector<double> p = R;
    std::array<double, 9> cell = box;
    for (std::size_t a = 0; a < R.size() / 3; ++a) {
      p[3 * a + col] += eps * R[3 * a + row];
    }
    for (int k = 0; k < 3; ++k) {
      cell[3 * k + col] += eps * box[3 * k + row];
    }
    return swStress(pot, p, cell).energy;
  };
  return (8.0 * (energyAt(h) - energyAt(-h)) -
          (energyAt(2 * h) - energyAt(-2 * h))) /
         (12.0 * h * rgpot::cellVolume(box.data()));
}

} // namespace

TEST_CASE("SW stress matches a strain difference and the 1985 kernel",
          "[fortran][sw][stress]") {
  rgpot::fortranpots::SWPot pot;
  REQUIRE(pot.caps().stress);
  // Both kernels are linear in epsilon and share every other parameter.
  const double ratio =
      2.16826 * 1.06767385 / rgpot::StillingerWeberSi::kEpsilon;
  const double l = 3 * 5.431;
  const std::array<double, 9> compressed{0.98 * l, 0, 0, 0, 0.98 * l,
                                         0,        0, 0, 0.98 * l};
  const std::array<double, 9> sheared{l, 0,        0,         0.10 * l, l,
                                      0, 0.05 * l, -0.07 * l, l};
  for (const auto &[box, wobble] :
       {std::pair{compressed, 0.0}, std::pair{sheared, 0.1}}) {
    const std::vector<double> R = siBlock(box, wobble);
    const SwStress got = swStress(pot, R, box);
    const SwStress ref = swStress(rgpot::StillingerWeberSi{}, R, box);
    REQUIRE_THAT(got.energy, WithinRel(ratio * ref.energy, 1e-13));
    for (int k = 0; k < 9; ++k) {
      CAPTURE(wobble, k, got.stress[k], ref.stress[k]);
      REQUIRE_THAT(got.stress[k], WithinAbs(ratio * ref.stress[k], 1e-14));
      REQUIRE_THAT(got.stress[k],
                   WithinAbs(swStrainDifference(R, box, k / 3, k % 3), 5e-10));
    }
    if (wobble == 0.0) {
      for (int k : {0, 4, 8}) {
        REQUIRE(got.stress[k] < -1e-2);
      }
    } else {
      for (int k : {1, 2, 5}) {
        REQUIRE(std::abs(got.stress[k]) > 1e-3);
      }
    }
  }
}

TEST_CASE("SW batch returns the forceImpl energies and forces without stress",
          "[fortran][sw][stress]") {
  rgpot::fortranpots::SWPot pot;
  const double l = 3 * 5.431;
  const std::array<double, 9> compressed{0.98 * l, 0, 0, 0, 0.98 * l,
                                         0,        0, 0, 0.98 * l};
  const std::array<double, 9> sheared{l, 0,        0,         0.10 * l, l,
                                      0, 0.05 * l, -0.07 * l, l};
  const std::array<std::array<double, 9>, 2> boxes{compressed, sheared};
  const std::array<std::vector<double>, 2> R{siBlock(compressed, 0.0),
                                             siBlock(sheared, 0.1)};
  const std::vector<int> types(R[0].size() / 3, 14);

  std::array<std::vector<double>, 2> refF, gotF;
  std::vector<rgpot::ForceInput> in;
  std::array<rgpot::ForceOut, 2> ref{}, got{};
  for (std::size_t s = 0; s < 2; ++s) {
    refF[s].assign(R[s].size(), 0.0);
    gotF[s].assign(R[s].size(), 0.0);
    in.push_back({.nAtoms = R[s].size() / 3,
                  .pos = R[s].data(),
                  .atmnrs = types.data(),
                  .box = boxes[s].data()});
    ref[s].F = refF[s].data();
    got[s].F = gotF[s].data();
    std::fill(got[s].stress, got[s].stress + 9, 1.0);
    got[s].has_stress = 1;
    pot.forceImpl(in[s], &ref[s]);
    REQUIRE(ref[s].has_stress == 1);
  }

  pot.forceBatch({.nSystems = 2, .in = in.data(), .out = got.data()});
  for (std::size_t s = 0; s < 2; ++s) {
    CAPTURE(s);
    REQUIRE(got[s].energy == ref[s].energy);
    REQUIRE(gotF[s] == refF[s]);
    REQUIRE(got[s].variance == 0.0);
    REQUIRE(got[s].has_stress == 0);
    for (const double v : got[s].stress) {
      REQUIRE(v == 0.0);
    }
  }
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

TEST_CASE("Tersoff dispatches silicon-carbide to the 1989 kernel",
          "[fortran][tersoff]") {
  const AtomMatrix pos{{0.0, 0.0, 0.0},
                       {2.30, 0.15, 0.05},
                       {0.40, 2.25, 0.20},
                       {1.15, 0.90, 1.40}};
  const std::vector<int> sic{14, 14, 14, 6};
  rgpot::fortranpots::TersoffPot face;
  rgpot::TersoffSiC kernel;
  auto [e_face, f_face, v_face] = face(pos, sic, cube20());
  auto [e_kern, f_kern, v_kern] = kernel(pos, sic, cube20());
  (void)v_face;
  (void)v_kern;
  REQUIRE_THAT(e_face, WithinAbs(-6.437125564429529, 1e-12));
  REQUIRE(e_face == e_kern);
  REQUIRE(f_face.size() == f_kern.size());
  for (size_t i = 0; i < f_face.size(); ++i) {
    REQUIRE(f_face.data()[i] == f_kern.data()[i]);
  }

  // The same coordinates with every Z = 14 stay on the silicon gather,
  // which does not reproduce the carbide energy.
  const std::vector<int> silicon(4, 14);
  auto [e_si, f_si, v_si] = face(pos, silicon, cube20());
  (void)f_si;
  (void)v_si;
  REQUIRE(e_si != e_face);

  std::vector<int> oxygen = silicon;
  oxygen[3] = 8;
  REQUIRE_THROWS_AS(face(pos, oxygen, cube20()), std::invalid_argument);
}

TEST_CASE("Tersoff null species stays on the silicon kernel",
          "[fortran][tersoff]") {
  rgpot::fortranpots::TersoffPot pot;
  const AtomMatrix pos = siDiamond();
  const std::vector<int> types(8, 14);
  auto [energy, forces, variance] = pot(pos, types, cube20());
  (void)variance;

  std::vector<double> got(pos.size(), 0.0);
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  rgpot::ForceInput in{.nAtoms = pos.rows(),
                       .pos = pos.data(),
                       .atmnrs = nullptr,
                       .box = box};
  rgpot::ForceOut fo{.F = got.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(in, &fo);
  REQUIRE(fo.energy == energy);
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE(got[i] == forces.data()[i]);
  }
}

TEST_CASE("Tersoff held skin reaches the 1989 kernel only",
          "[fortran][tersoff]") {
  AtomMatrix pos{{0.0, 0.0, 0.0},
                 {2.30, 0.15, 0.05},
                 {0.40, 2.25, 0.20},
                 {1.15, 0.90, 1.40}};
  const std::vector<int> sic{14, 14, 14, 6};
  rgpot::fortranpots::TersoffPot plain;
  rgpot::fortranpots::TersoffPot held;
  held.setHeldSkin(0.4);
  REQUIRE(held.caps().reentrancy == rgpot::Reentrancy::PerInstance);
  for (int step = 0; step < 3; ++step) {
    auto [e_plain, f_plain, v_plain] = plain(pos, sic, cube20());
    auto [e_held, f_held, v_held] = held(pos, sic, cube20());
    (void)v_plain;
    (void)v_held;
    REQUIRE(e_held == e_plain);
    for (size_t i = 0; i < f_plain.size(); ++i) {
      REQUIRE(f_held.data()[i] == f_plain.data()[i]);
    }
    pos(1, 0) += 0.05;
  }

  const std::vector<int> silicon(8, 14);
  auto [e_plain, f_plain, v_plain] = plain(siDiamond(), silicon, cube20());
  auto [e_held, f_held, v_held] = held(siDiamond(), silicon, cube20());
  (void)v_plain;
  (void)v_held;
  REQUIRE(e_held == e_plain);
  for (size_t i = 0; i < f_plain.size(); ++i) {
    REQUIRE(f_held.data()[i] == f_plain.data()[i]);
  }
  REQUIRE_THROWS_AS(held.setHeldSkin(-1.0), std::invalid_argument);
}

namespace {

struct TersoffStress {
  double energy = 0.0;
  std::array<double, 9> stress{};
  std::vector<double> force;
};

TersoffStress tersoffStress(const rgpot::fortranpots::TersoffPot &pot,
                            const std::vector<double> &R,
                            const std::vector<int> &types,
                            const std::array<double, 9> &box) {
  TersoffStress out;
  out.force.assign(R.size(), 0.0);
  rgpot::ForceInput in{.nAtoms = R.size() / 3,
                       .pos = R.data(),
                       .atmnrs = types.data(),
                       .box = box.data()};
  rgpot::ForceOut fo{.F = out.force.data(),
                     .energy = 0.0,
                     .variance = 0.0,
                     .stress = {},
                     .has_stress = 0};
  pot.forceImpl(in, &fo);
  REQUIRE(fo.has_stress == 1);
  out.energy = fo.energy;
  std::copy(fo.stress, fo.stress + 9, out.stress.begin());
  return out;
}

double tersoffStrainDifference(const std::vector<double> &R,
                               const std::vector<int> &types,
                               const std::array<double, 9> &box, int row,
                               int col) {
  constexpr double h = 2e-4;
  const rgpot::fortranpots::TersoffPot pot;
  auto energyAt = [&](double eps) {
    std::vector<double> p = R;
    std::array<double, 9> cell = box;
    for (std::size_t a = 0; a < R.size() / 3; ++a) {
      p[3 * a + col] += eps * R[3 * a + row];
    }
    for (int k = 0; k < 3; ++k) {
      cell[3 * k + col] += eps * box[3 * k + row];
    }
    return tersoffStress(pot, p, types, cell).energy;
  };
  return (8.0 * (energyAt(h) - energyAt(-h)) -
          (energyAt(2 * h) - energyAt(-2 * h))) /
         (12.0 * h * rgpot::cellVolume(box.data()));
}

void checkSymmetric(const std::array<double, 9> &s) {
  REQUIRE(s[1] == s[3]);
  REQUIRE(s[2] == s[6]);
  REQUIRE(s[5] == s[7]);
}

} // namespace

TEST_CASE("Tersoff wrapper stress matches a strain difference on gather path",
          "[fortran][tersoff][stress]") {
  rgpot::fortranpots::TersoffPot pot;
  REQUIRE(pot.caps().stress);
  const double a = 5.432;
  std::vector<double> diamond;
  rgpot::test_fixtures::appendDiamond(diamond, 2, a);
  const std::vector<int> si(diamond.size() / 3, 14);

  SECTION("periodic diamond") {
    const std::array<double, 9> box{2 * a, 0, 0, 0, 2 * a, 0, 0, 0, 2 * a};
    const TersoffStress got = tersoffStress(pot, diamond, si, box);
    checkSymmetric(got.stress);
    for (int k = 0; k < 9; ++k) {
      CAPTURE(k, got.stress[k]);
      REQUIRE_THAT(got.stress[k],
                   WithinAbs(tersoffStrainDifference(diamond, si, box, k / 3,
                                                     k % 3),
                             5e-10));
    }
  }

  SECTION("strained rattled silicon cell") {
    const std::array<double, 9> box{2 * a, 0, 0, 0.06 * a, 2 * a,
                                    0, -0.03 * a, 0.04 * a, 2 * a};
    std::vector<double> rattled = diamond;
    std::mt19937_64 gen(20261123);
    for (double &x : rattled) {
      x += 0.3 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
    }
    const TersoffStress got = tersoffStress(pot, rattled, si, box);
    checkSymmetric(got.stress);
    for (int k = 0; k < 9; ++k) {
      CAPTURE(k, got.stress[k]);
      REQUIRE_THAT(got.stress[k],
                   WithinAbs(tersoffStrainDifference(rattled, si, box, k / 3,
                                                     k % 3),
                             5e-10));
    }
  }
}

TEST_CASE("Tersoff wrapper stress matches SiC kernel on mixed species path",
          "[fortran][tersoff][stress]") {
  std::vector<double> pos;
  rgpot::test_fixtures::appendDiamond(pos, 2, 4.03);
  std::vector<int> types(pos.size() / 3, 14);
  for (std::size_t a = 0; a < types.size(); ++a) {
    if (a % 8 < 4) {
      types[a] = 6;
    }
  }
  std::mt19937_64 gen(20261123);
  for (double &x : pos) {
    x += 0.05 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
  }
  const std::array<double, 9> box{8.06, 0, 0, 0.18, 8.06, 0, 0, -0.11, 8.06};
  rgpot::fortranpots::TersoffPot face;
  rgpot::TersoffSiC kernel;
  std::vector<double> ff(pos.size(), 0.0);
  std::vector<double> fk(pos.size(), 0.0);
  rgpot::ForceInput in{
      .nAtoms = pos.size() / 3, .pos = pos.data(), .atmnrs = types.data(), .box = box.data()};
  rgpot::ForceOut of{
      .F = ff.data(), .energy = 0.0, .variance = 0.0, .stress = {}, .has_stress = 0};
  rgpot::ForceOut ok{
      .F = fk.data(), .energy = 0.0, .variance = 0.0, .stress = {}, .has_stress = 0};
  face.forceImpl(in, &of);
  kernel.forceImpl(in, &ok);
  REQUIRE(of.has_stress == 1);
  REQUIRE(ok.has_stress == 1);
  REQUIRE(of.energy == ok.energy);
  REQUIRE(of.variance == ok.variance);
  REQUIRE(ff == fk);
  for (int k = 0; k < 9; ++k) {
    REQUIRE(of.stress[k] == ok.stress[k]);
  }
  checkSymmetric({of.stress[0], of.stress[1], of.stress[2], of.stress[3],
                  of.stress[4], of.stress[5], of.stress[6], of.stress[7],
                  of.stress[8]});
}

TEST_CASE("Tersoff gather stress is invariant across thread counts",
          "[fortran][tersoff][stress]") {
#ifdef _OPENMP
  std::vector<double> pos;
  rgpot::test_fixtures::appendDiamond(pos, 2, 5.432);
  std::vector<int> types(pos.size() / 3, 14);
  const std::array<double, 9> box{10.864, 0, 0, 0.2, 10.864, 0, 0, 0.15,
                                  10.864};
  rgpot::fortranpots::TersoffPot pot;
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1);
  const TersoffStress one = tersoffStress(pot, pos, types, box);
  const int more = std::min(saved > 1 ? saved : 2, 4);
  omp_set_num_threads(more);
  const TersoffStress many = tersoffStress(pot, pos, types, box);
  omp_set_num_threads(saved);
  REQUIRE(one.energy == many.energy);
  REQUIRE(one.force == many.force);
  REQUIRE(one.stress == many.stress);
#else
  SUCCEED("OpenMP not enabled in this build");
#endif
}

TEST_CASE("Tersoff no-box zero-atom call keeps has_stress clear",
          "[fortran][tersoff][stress]") {
  rgpot::fortranpots::TersoffPot pot;
  double force = 1.0;
  rgpot::ForceInput in{
      .nAtoms = 0, .pos = nullptr, .atmnrs = nullptr, .box = nullptr};
  rgpot::ForceOut out{
      .F = &force, .energy = 1.0, .variance = 1.0, .stress = {1, 2, 3, 4, 5, 6, 7, 8, 9}, .has_stress = 1};
  pot.forceImpl(in, &out);
  REQUIRE(out.energy == 0.0);
  REQUIRE(out.variance == 0.0);
  REQUIRE(out.has_stress == 0);
  for (double c : out.stress) {
    REQUIRE(c == 0.0);
  }

  const AtomMatrix pos = siDiamond();
  const std::vector<int> types(pos.rows(), 14);
  std::vector<double> force_buf(pos.size(), 0.0);
  rgpot::ForceInput atom_in{
      .nAtoms = pos.rows(),
      .pos = pos.data(),
      .atmnrs = types.data(),
      .box = nullptr};
  rgpot::ForceOut atom_out{.F = force_buf.data(),
                           .energy = 1.0,
                           .variance = 1.0,
                           .stress = {1, 2, 3, 4, 5, 6, 7, 8, 9},
                           .has_stress = 1};
  REQUIRE_THROWS_WITH(pot.forceImpl(atom_in, &atom_out),
                      ContainsSubstring("required neighbour pointer is null"));
}

TEST_CASE("Tersoff negative-determinant box matches right-handed mirror",
          "[fortran][tersoff][stress]") {
  rgpot::fortranpots::TersoffPot pot;
  const double a = 5.432;
  std::vector<double> diamond;
  rgpot::test_fixtures::appendDiamond(diamond, 2, a);
  std::mt19937_64 gen(20261124);
  for (double &x : diamond) {
    x += 0.08 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
  }
  const std::array<double, 9> box_rh{
      2 * a, 0, 0, 0.06 * a, 2 * a, 0, -0.03 * a, 0.04 * a, 2 * a};

  auto checkMirror = [&](const std::vector<int> &types) {
    const TersoffStress rh = tersoffStress(pot, diamond, types, box_rh);

    // Negating one lattice vector flips det(box) < 0 without changing the
    // Bravais lattice or atomic positions.
    std::array<double, 9> box_flip_a = box_rh;
    box_flip_a[0] = -box_flip_a[0];
    box_flip_a[1] = -box_flip_a[1];
    box_flip_a[2] = -box_flip_a[2];
    const TersoffStress lh_vec = tersoffStress(pot, diamond, types, box_flip_a);
    REQUIRE_THAT(lh_vec.energy, WithinAbs(rh.energy, 1e-12));
    for (std::size_t i = 0; i < rh.force.size(); ++i) {
      REQUIRE_THAT(lh_vec.force[i], WithinAbs(rh.force[i], 1e-12));
    }
    for (int k = 0; k < 9; ++k) {
      REQUIRE_THAT(lh_vec.stress[k], WithinAbs(rh.stress[k], 1e-12));
    }

    // Spatial inversion (pos, box) -> (-pos, -box) has det < 0 and keeps
    // energy and the entire stress tensor invariant.
    std::vector<double> pos_inv(diamond.size());
    for (std::size_t i = 0; i < diamond.size(); ++i) {
      pos_inv[i] = -diamond[i];
    }
    std::array<double, 9> box_inv{};
    for (int k = 0; k < 9; ++k) {
      box_inv[k] = -box_rh[k];
    }
    const TersoffStress lh_inv = tersoffStress(pot, pos_inv, types, box_inv);
    REQUIRE_THAT(lh_inv.energy, WithinAbs(rh.energy, 1e-12));
    for (std::size_t i = 0; i < rh.force.size(); ++i) {
      REQUIRE_THAT(lh_inv.force[i], WithinAbs(-rh.force[i], 1e-12));
    }
    for (int k = 0; k < 9; ++k) {
      REQUIRE_THAT(lh_inv.stress[k], WithinAbs(rh.stress[k], 1e-12));
    }

    // Single-axis mirror z -> -z (det < 0) keeps energy invariant and
    // transforms forces as P*f and stress as P*sigma*P^T.
    std::vector<double> pos_z(diamond);
    for (std::size_t i = 2; i < pos_z.size(); i += 3) {
      pos_z[i] = -pos_z[i];
    }
    std::array<double, 9> box_z = box_rh;
    for (int k = 2; k < 9; k += 3) {
      box_z[k] = -box_z[k];
    }
    const TersoffStress lh_z = tersoffStress(pot, pos_z, types, box_z);
    REQUIRE_THAT(lh_z.energy, WithinAbs(rh.energy, 1e-12));
    for (std::size_t i = 0; i < rh.force.size(); ++i) {
      const double sign = (i % 3 == 2) ? -1.0 : 1.0;
      REQUIRE_THAT(lh_z.force[i], WithinAbs(sign * rh.force[i], 1e-12));
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        const double sr = (r == 2) ? -1.0 : 1.0;
        const double sc = (c == 2) ? -1.0 : 1.0;
        REQUIRE_THAT(lh_z.stress[3 * r + c],
                     WithinAbs(sr * sc * rh.stress[3 * r + c], 1e-12));
      }
    }
  };

  SECTION("all-silicon gather path") {
    const std::vector<int> si(diamond.size() / 3, 14);
    checkMirror(si);
  }

  SECTION("mixed-species SiC path") {
    std::vector<int> sic(diamond.size() / 3, 14);
    for (std::size_t i = 0; i < sic.size(); ++i) {
      if (i % 8 < 4) {
        sic[i] = 6;
      }
    }
    checkMirror(sic);
  }

  SECTION("zero-volume orthorhombic box") {
    const std::array<double, 9> zero_ortho{10.0, 0, 0, 0, 0.0, 0, 0, 0, 10.0};
    const std::vector<int> si(diamond.size() / 3, 14);
    std::vector<double> force_buf(diamond.size(), 0.0);
    rgpot::ForceInput si_in{.nAtoms = si.size(),
                            .pos = diamond.data(),
                            .atmnrs = si.data(),
                            .box = zero_ortho.data()};
    rgpot::ForceOut si_out{.F = force_buf.data(),
                           .energy = 0.0,
                           .variance = 0.0,
                           .stress = {},
                           .has_stress = 0};
    REQUIRE_THROWS_WITH(pot.forceImpl(si_in, &si_out),
                        ContainsSubstring("positive volume"));

    std::vector<int> sic(si);
    sic[0] = 6;
    rgpot::ForceInput sic_in{.nAtoms = sic.size(),
                             .pos = diamond.data(),
                             .atmnrs = sic.data(),
                             .box = zero_ortho.data()};
    rgpot::ForceOut sic_out{.F = force_buf.data(),
                            .energy = 0.0,
                            .variance = 0.0,
                            .stress = {1, 2, 3, 4, 5, 6, 7, 8, 9},
                            .has_stress = 1};
    pot.forceImpl(sic_in, &sic_out);
    REQUIRE(sic_out.has_stress == 0);
    for (double c : sic_out.stress) {
      REQUIRE(c == 0.0);
    }
  }
}

TEST_CASE("Tersoff concurrent errors keep complete messages",
          "[fortran][tersoff][threads]") {
  constexpr int threadCount = 16;
  constexpr int iterations = 100;
  const AtomMatrix pos = siDiamond();
  const std::vector<int> types(pos.rows(), 14);
  const std::array<double, 9> zeroBox{10.0, 0, 0, 0, 0.0, 0, 0, 0, 10.0};
  const std::string expected =
      "Tersoff potential failed (status 4): rgpot_neighbors: box must be "
      "finite, and a periodic box must have positive volume";
  std::barrier<> ready(threadCount);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (int worker = 0; worker < threadCount; ++worker) {
    threads.emplace_back([&] {
      rgpot::fortranpots::TersoffPot pot;
      std::vector<double> force(pos.size(), 0.0);
      rgpot::ForceInput in{.nAtoms = pos.rows(),
                           .pos = pos.data(),
                           .atmnrs = types.data(),
                           .box = zeroBox.data()};
      for (int attempt = 0; attempt < iterations; ++attempt) {
        rgpot::ForceOut out{.F = force.data(),
                            .energy = 0.0,
                            .variance = 0.0,
                            .stress = {},
                            .has_stress = 0};
        ready.arrive_and_wait();
        try {
          pot.forceImpl(in, &out);
          ++failures;
        } catch (const std::runtime_error &error) {
          if (error.what() != expected) {
            ++failures;
          }
        } catch (...) {
          ++failures;
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  REQUIRE(failures.load() == 0);
}

TEST_CASE("Tersoff stress path from many threads matches one thread",
          "[fortran][tersoff][stress][threads]") {
  constexpr int threadCount = 12;
  constexpr int rounds = 20;
  struct Frame {
    std::vector<double> pos;
    std::vector<int> types;
    std::array<double, 9> box{};
  };
  std::vector<Frame> frames(threadCount);
  for (int k = 0; k < threadCount; ++k) {
    Frame &frame = frames[static_cast<std::size_t>(k)];
    const bool silicon = k % 2 == 0;
    const double a = silicon ? 5.432 : 4.03;
    rgpot::test_fixtures::appendDiamond(frame.pos, 2, a);
    frame.types.assign(frame.pos.size() / 3, 14);
    if (!silicon) {
      for (std::size_t i = 0; i < frame.types.size(); ++i) {
        if (i % 8 < 4) {
          frame.types[i] = 6;
        }
      }
    }
    std::mt19937_64 gen(20261125 + static_cast<unsigned>(k));
    for (double &x : frame.pos) {
      x += 0.08 * (static_cast<double>(gen() >> 11) * 0x1.0p-53 - 0.5);
    }
    frame.box = {2 * a, 0, 0, 0.03 * k, 2 * a, 0, 0, -0.02 * k, 2 * a};
  }
  const std::array<double, 9> zeroBox{10.0, 0, 0, 0, 0.0, 0, 0, 0, 10.0};
  const std::string expected =
      "Tersoff potential failed (status 4): rgpot_neighbors: box must be "
      "finite, and a periodic box must have positive volume";

  std::vector<TersoffStress> refs;
  {
    rgpot::fortranpots::TersoffPot pot;
    for (const Frame &frame : frames) {
      refs.push_back(tersoffStress(pot, frame.pos, frame.types, frame.box));
    }
  }

  std::barrier<> ready(threadCount);
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(threadCount);
  for (int worker = 0; worker < threadCount; ++worker) {
    threads.emplace_back([&, worker] {
      const Frame &frame = frames[static_cast<std::size_t>(worker)];
      const Frame &silicon = frames[static_cast<std::size_t>(worker - worker % 2)];
      const TersoffStress &ref = refs[static_cast<std::size_t>(worker)];
      rgpot::fortranpots::TersoffPot pot;
      std::vector<double> force(frame.pos.size(), 0.0);
      rgpot::ForceInput in{.nAtoms = frame.pos.size() / 3,
                           .pos = frame.pos.data(),
                           .atmnrs = frame.types.data(),
                           .box = frame.box.data()};
      rgpot::ForceInput bad{.nAtoms = silicon.pos.size() / 3,
                            .pos = silicon.pos.data(),
                            .atmnrs = silicon.types.data(),
                            .box = zeroBox.data()};
      for (int round = 0; round < rounds; ++round) {
        rgpot::ForceOut out{.F = force.data(),
                            .energy = 0.0,
                            .variance = 0.0,
                            .stress = {},
                            .has_stress = 0};
        ready.arrive_and_wait();
        try {
          pot.forceImpl(in, &out);
          if (out.has_stress != 1 || out.energy != ref.energy ||
              force != ref.force ||
              !std::equal(ref.stress.begin(), ref.stress.end(), out.stress)) {
            ++failures;
          }
        } catch (...) {
          ++failures;
        }
        ready.arrive_and_wait();
        try {
          pot.forceImpl(bad, &out);
          ++failures;
        } catch (const std::runtime_error &error) {
          if (error.what() != expected) {
            ++failures;
          }
        } catch (...) {
          ++failures;
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  REQUIRE(failures.load() == 0);
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

namespace {

/// Diamond supercell `nx` by `ny` by `nz`, each atom moved by `amp`.
std::vector<double> edipDiamond(int nx, int ny, int nz, double a, double amp) {
  static const double basis[8][3] = {
      {0.00, 0.00, 0.00}, {0.00, 0.50, 0.50}, {0.50, 0.00, 0.50},
      {0.50, 0.50, 0.00}, {0.25, 0.25, 0.25}, {0.25, 0.75, 0.75},
      {0.75, 0.25, 0.75}, {0.75, 0.75, 0.25}};
  std::vector<double> pos;
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        for (const auto &b : basis) {
          const auto k = static_cast<double>(pos.size() / 3);
          pos.push_back((ix + b[0]) * a + amp * std::sin(1.3 * k));
          pos.push_back((iy + b[1]) * a + amp * std::sin(1.7 * k + 0.4));
          pos.push_back((iz + b[2]) * a + amp * std::sin(2.1 * k + 0.8));
        }
      }
    }
  }
  return pos;
}

double bitsToDouble(std::uint64_t bits) {
  double value = 0.0;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// (1/V) dE/d(eps_row,col) by the fourth-order central difference, step
/// 2e-4. The strain maps every position and cell row r to r (I + eps).
double edipStrainDifference(const std::vector<double> &R,
                            const std::array<double, 9> &box, int row,
                            int col) {
  constexpr double h = 2e-4;
  const rgpot::fortranpots::EDIPPot pot;
  auto energyAt = [&](double eps) {
    std::vector<double> p = R;
    std::array<double, 9> cell = box;
    for (std::size_t a = 0; a < R.size() / 3; ++a) {
      p[3 * a + col] += eps * R[3 * a + row];
    }
    for (int k = 0; k < 3; ++k) {
      cell[3 * k + col] += eps * box[3 * k + row];
    }
    return swStress(pot, p, cell).energy;
  };
  return (8.0 * (energyAt(h) - energyAt(-h)) -
          (energyAt(2 * h) - energyAt(-2 * h))) /
         (12.0 * h * rgpot::cellVolume(box.data()));
}

} // namespace

TEST_CASE("EDIP stress matches a strain difference", "[fortran][edip][stress]") {
  rgpot::fortranpots::EDIPPot pot;
  REQUIRE(pot.caps().stress);
  // 216 atoms, the cell the Stillinger-Weber stress test uses. Both edges
  // stay well above the EDIP cutoff.
  const double l = 3 * 5.431;
  const std::array<double, 9> compressed{0.98 * l, 0, 0, 0, 0.98 * l,
                                         0,        0, 0, 0.98 * l};
  const std::array<double, 9> sheared{l, 0,        0,         0.10 * l, l,
                                      0, 0.05 * l, -0.07 * l, l};
  for (const auto &[box, wobble] :
       {std::pair{compressed, 0.0}, std::pair{sheared, 0.1}}) {
    const std::vector<double> R = siBlock(box, wobble);
    const SwStress got = swStress(pot, R, box);
    REQUIRE(got.stress[1] == got.stress[3]);
    REQUIRE(got.stress[2] == got.stress[6]);
    REQUIRE(got.stress[5] == got.stress[7]);
    for (int k = 0; k < 9; ++k) {
      CAPTURE(wobble, k, got.stress[k]);
      REQUIRE_THAT(got.stress[k],
                   WithinAbs(edipStrainDifference(R, box, k / 3, k % 3), 5e-10));
    }
    if (wobble == 0.0) {
      for (int k : {0, 4, 8}) {
        REQUIRE(got.stress[k] < -1e-2);
      }
    } else {
      for (int k : {1, 2, 5}) {
        REQUIRE(std::abs(got.stress[k]) > 1e-3);
      }
    }
  }
}

TEST_CASE("EDIP energies and forces stay at the recorded bits",
          "[fortran][edip][stress]") {
  // 4 x 4 x 2 conventional cells, 256 atoms, rattled by 0.05 Angstrom.
  // Recorded from the force path before the virial was added.
  const double a = 5.431;
  const double box[9] = {4 * a, 0, 0, 0, 4 * a, 0, 0, 0, 2 * a};
  const std::vector<double> pos = edipDiamond(4, 4, 2, a, 0.05);
  std::vector<double> forces(pos.size(), 0.0);
  const std::vector<int> types(pos.size() / 3, 14);
  rgpot::ForceOut out{.F = forces.data(),
                      .energy = 0.0,
                      .variance = 0.0,
                      .stress = {},
                      .has_stress = 0};
  rgpot::fortranpots::EDIPPot{}.forceImpl(
      {pos.size() / 3, pos.data(), types.data(), box}, &out);
  REQUIRE(out.has_stress == 1);
  REQUIRE(out.energy == bitsToDouble(0xc0927763b2663a0aULL));
  REQUIRE(forces[0] == bitsToDouble(0xbfbb1698545b0c6fULL));
}

TEST_CASE("EDIP stress is zero for an isolated atom", "[fortran][edip][stress]") {
  const double position[3] = {1.0, 2.0, 3.0};
  const int type = 14;
  const double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  double forces[3] = {};
  rgpot::ForceOut out{forces, 1.0, 0.0, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0};
  rgpot::fortranpots::EDIPPot{}.forceImpl({1, position, &type, box}, &out);
  REQUIRE(out.has_stress == 1);
  REQUIRE(out.energy == 0.0);
  for (double component : out.stress) {
    REQUIRE(component == 0.0);
  }
  for (double component : forces) {
    REQUIRE(component == 0.0);
  }
}

TEST_CASE("EDIP batch returns the forceImpl energies and forces without stress",
          "[fortran][edip][stress]") {
  rgpot::fortranpots::EDIPPot pot;
  const double l = 3 * 5.431;
  const std::array<double, 9> compressed{0.98 * l, 0, 0, 0, 0.98 * l,
                                         0,        0, 0, 0.98 * l};
  const std::array<double, 9> sheared{l, 0,        0,         0.10 * l, l,
                                      0, 0.05 * l, -0.07 * l, l};
  const std::array<std::array<double, 9>, 2> boxes{compressed, sheared};
  const std::array<std::vector<double>, 2> R{siBlock(compressed, 0.0),
                                             siBlock(sheared, 0.1)};
  const std::vector<int> types(R[0].size() / 3, 14);

  std::array<std::vector<double>, 2> refF, gotF;
  std::vector<rgpot::ForceInput> in;
  std::array<rgpot::ForceOut, 2> ref{}, got{};
  for (std::size_t s = 0; s < 2; ++s) {
    refF[s].assign(R[s].size(), 0.0);
    gotF[s].assign(R[s].size(), 0.0);
    in.push_back({.nAtoms = R[s].size() / 3,
                  .pos = R[s].data(),
                  .atmnrs = types.data(),
                  .box = boxes[s].data()});
    ref[s].F = refF[s].data();
    got[s].F = gotF[s].data();
    std::fill(got[s].stress, got[s].stress + 9, 1.0);
    got[s].has_stress = 1;
    pot.forceImpl(in[s], &ref[s]);
    REQUIRE(ref[s].has_stress == 1);
  }

  pot.forceBatch({.nSystems = 2, .in = in.data(), .out = got.data()});
  for (std::size_t s = 0; s < 2; ++s) {
    CAPTURE(s);
    REQUIRE(std::memcmp(&got[s].energy, &ref[s].energy, sizeof(double)) == 0);
    REQUIRE(std::memcmp(gotF[s].data(), refF[s].data(),
                        gotF[s].size() * sizeof(double)) == 0);
    REQUIRE(got[s].variance == 0.0);
    REQUIRE(got[s].has_stress == 0);
    for (const double v : got[s].stress) {
      REQUIRE(v == 0.0);
    }
  }
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
