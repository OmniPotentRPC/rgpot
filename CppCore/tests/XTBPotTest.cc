// MIT License
// Copyright 2023--present rgpot developers
//
// libxtb is Fortran with an ISO_C_BINDING C API (xtb.h). Unit tests exercise
// that C contract carefully:
//   - opaque handle create / destroy (xtb_new* / xtb_del*)
//   - xtb_releaseOutput on each environment
//   - first force: xtb_newMolecule + loadGFN*; later: xtb_updateMolecule
//   - serial multi-handle lifecycle (independent ISO_C handle sets)
//   - method switch only after full destroy of previous handles
//   - linked XTBPot vs dlopen libxtb_engine (same C semantics)
// Tags: [xtb][capi] [xtb][linked] [xtb][dlopen]

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "rgpot/XTBPot/XTBDlopen.hpp"
#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

// Water (Å): O + 2H
static const double kWaterPos[] = {
    0.00000000, 0.00000000,  0.11779000,  // O
    0.00000000, 0.75545000,  -0.47116000, // H
    0.00000000, -0.75545000, -0.47116000  // H
};
static const int kWaterZ[] = {8, 1, 1};

static void fill_water(AtomMatrix &positions, std::vector<int> &atmtypes,
                       std::array<std::array<double, 3>, 3> &box) {
  // Caller constructs positions(3,3).
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      positions(static_cast<size_t>(i), static_cast<size_t>(j)) =
          kWaterPos[i * 3 + j];
  atmtypes.assign(kWaterZ, kWaterZ + 3);
  box = {{{100.0, 0.0, 0.0}, {0.0, 100.0, 0.0}, {0.0, 0.0, 100.0}}};
}

static void displace_water(AtomMatrix &positions, double delta) {
  // Break exact symmetry slightly; still a valid bound geometry.
  positions(1, 1) += delta;
  positions(2, 1) -= delta;
}

static void require_finite_forces(const AtomMatrix &forces, size_t n) {
  REQUIRE(forces.rows() == n);
  REQUIRE(forces.cols() == 3);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(std::isfinite(forces(i, j)));
}

static void require_zero_net_force(const AtomMatrix &forces, double tol) {
  double fx = 0, fy = 0, fz = 0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    fx += forces(i, 0);
    fy += forces(i, 1);
    fz += forces(i, 2);
  }
  REQUIRE_THAT(fx, WithinAbs(0.0, tol));
  REQUIRE_THAT(fy, WithinAbs(0.0, tol));
  REQUIRE_THAT(fz, WithinAbs(0.0, tol));
}

static rgpot::XTBDlopenConfig dlopen_cfg(const rgpot::XTBConfig &xtb) {
  rgpot::XTBDlopenConfig c;
  c.xtb = xtb;
  if (const char *eng = std::getenv("RGPOT_XTB_ENGINE"))
    if (eng && *eng)
      c.engine_path = eng;
  return c;
}

// ---------------------------------------------------------------------------
// Linked XTBPot (build-time libxtb)
// ---------------------------------------------------------------------------

TEST_CASE("XTBPot GFN2 water energy and force conservation", "[xtb][linked]") {
  rgpot::XTBPot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;

  REQUIRE(std::isfinite(energy));
  // GFN2-xTB water ≈ -5.07 Ha ≈ -137.9 eV
  REQUIRE(energy < 0.0);
  REQUIRE_THAT(energy, WithinAbs(-137.9, 1.0));
  require_finite_forces(forces, 3);
  require_zero_net_force(forces, 1e-6);
}

TEST_CASE("XTBPot GFNFF water finite energy", "[xtb][linked][gfnff]") {
  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFNFF;
  rgpot::XTBPot pot(cfg);
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [energy, forces, variance] = pot(positions, atmtypes, box);
  (void)variance;
  REQUIRE(std::isfinite(energy));
  REQUIRE(energy < 0.0);
  require_finite_forces(forces, 3);
}

TEST_CASE("XTBPot warm updateMolecule path is stable", "[xtb][linked][capi]") {
  // ISO_C: after first force, molecule handle is updated via xtb_updateMolecule.
  rgpot::XTBPot pot;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  auto [e0, f0, v0] = pot(positions, atmtypes, box);
  (void)v0;
  REQUIRE(std::isfinite(e0));

  AtomMatrix moved(3, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      moved(i, j) = positions(i, j);
  displace_water(moved, 0.02);
  auto [e1, f1, v1] = pot(moved, atmtypes, box);
  (void)v1;
  REQUIRE(std::isfinite(e1));
  REQUIRE(std::abs(e1 - e0) > 1e-6); // geometry change must move energy
  require_finite_forces(f1, 3);

  // Return to original geometry: energy must recover (warm SCF state).
  auto [e2, f2, v2] = pot(positions, atmtypes, box);
  (void)f0;
  (void)f2;
  (void)v2;
  REQUIRE_THAT(e2, WithinAbs(e0, 1e-8));
}

TEST_CASE("XTBPot serial multi-instance lifecycle (ISO_C handles)",
          "[xtb][linked][capi]") {
  // Serial multi-handle lifecycle (independent ISO_C environments).
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  double e_ref = 0.0;
  {
    rgpot::XTBPot pot;
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)f;
    (void)v;
    e_ref = e;
  }

  constexpr int kRounds = 8;
  for (int r = 0; r < kRounds; ++r) {
    rgpot::XTBPot pot;
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)v;
    REQUIRE(std::isfinite(e));
    REQUIRE_THAT(e, WithinAbs(e_ref, 1e-8));
    require_finite_forces(f, 3);
    require_zero_net_force(f, 1e-5);
  } // destructor: xtb_del* for this handle set
}

TEST_CASE("XTBPot sequential method switch after destroy",
          "[xtb][linked][capi]") {
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  const rgpot::GFNMethod methods[] = {
      rgpot::GFNMethod::GFN2xTB,
      rgpot::GFNMethod::GFNFF,
      rgpot::GFNMethod::GFN1xTB,
      rgpot::GFNMethod::GFN2xTB,
  };
  for (auto m : methods) {
    rgpot::XTBConfig cfg;
    cfg.method = m;
    rgpot::XTBPot pot(cfg);
    auto [e, f, v] = pot(positions, atmtypes, box);
    (void)v;
    REQUIRE(std::isfinite(e));
    REQUIRE(e < 0.0);
    require_finite_forces(f, 3);
  }
}

// ---------------------------------------------------------------------------
// dlopen libxtb_engine.so (Fortran still inside the plugin)
// ---------------------------------------------------------------------------

TEST_CASE("XTBDlopen matches linked GFN2 water", "[xtb][dlopen][capi]") {
  const char *eng = std::getenv("RGPOT_XTB_ENGINE");
  if (!eng || !*eng) {
    WARN("RGPOT_XTB_ENGINE unset; attempting bare libxtb_engine.so");
  }

  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFN2xTB;

  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  // Parity: evaluate linked to completion (destroy handles), then dlopen engine.
  double e_l = 0.0;
  AtomMatrix f_l(3, 3);
  {
    rgpot::XTBPot linked(cfg);
    auto [e, f, v] = linked(positions, atmtypes, box);
    (void)v;
    e_l = e;
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        f_l(i, j) = f(i, j);
    auto [e2, f2, v2] = linked(positions, atmtypes, box);
    (void)f2;
    (void)v2;
    REQUIRE_THAT(e2, WithinAbs(e_l, 1e-8));
  } // linked Fortran env destroyed

  std::unique_ptr<rgpot::XTBDlopen> dl;
  try {
    dl = std::make_unique<rgpot::XTBDlopen>(dlopen_cfg(cfg));
  } catch (const std::exception &ex) {
    FAIL("XTBDlopen construct failed: " << ex.what());
  }
  REQUIRE(dl->available());

  auto [e_d, f_d, v_d] = (*dl)(positions, atmtypes, box);
  (void)v_d;
  REQUIRE(std::isfinite(e_d));
  REQUIRE_THAT(e_d, WithinAbs(e_l, 1e-8));
  require_finite_forces(f_d, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE_THAT(f_d(i, j), WithinAbs(f_l(i, j), 1e-6));

  // Warm update path inside the engine (xtb_updateMolecule)
  AtomMatrix moved(3, 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      moved(i, j) = positions(i, j);
  displace_water(moved, 0.015);
  auto [e_m, f_m, v_m] = (*dl)(moved, atmtypes, box);
  (void)f_m;
  (void)v_m;
  REQUIRE(std::isfinite(e_m));
  auto [e_back, f_back, v_back] = (*dl)(positions, atmtypes, box);
  (void)f_back;
  (void)v_back;
  REQUIRE_THAT(e_back, WithinAbs(e_d, 1e-8));
}

TEST_CASE("XTBDlopen serial multi-instance via recreate",
          "[xtb][dlopen][capi]") {
  // Destroy and recreate plugin pot (new C ABI pot → new ISO_C handle set).
  rgpot::XTBConfig cfg;
  AtomMatrix positions(3, 3);
  std::vector<int> atmtypes;
  std::array<std::array<double, 3>, 3> box{};
  fill_water(positions, atmtypes, box);

  double e_ref = 0.0;
  {
    rgpot::XTBDlopen dl(dlopen_cfg(cfg));
    auto [e, f, v] = dl(positions, atmtypes, box);
    (void)f;
    (void)v;
    e_ref = e;
  }

  for (int r = 0; r < 4; ++r) {
    rgpot::XTBDlopen dl(dlopen_cfg(cfg));
    auto [e, f, v] = dl(positions, atmtypes, box);
    (void)v;
    REQUIRE_THAT(e, WithinAbs(e_ref, 1e-8));
    require_finite_forces(f, 3);
  }
}

// ---------------------------------------------------------------------------
// Periodic boundary conditions derived from the box
// ---------------------------------------------------------------------------

namespace {

// Water in a 6 A cube with the second hydrogen written as its image one
// lattice vector away; a periodic evaluation cannot tell the two apart.
constexpr double kPeriodicEdge = 6.0;

// GFN2-xTB runs isolated; GFN1-xTB is the periodic method of libxtb.
rgpot::XTBConfig periodicConfig() {
  rgpot::XTBConfig cfg;
  cfg.method = rgpot::GFNMethod::GFN1xTB;
  return cfg;
}

struct PeriodicResult {
  double energy;
  int has_stress;
  std::array<double, 9> stress;
};

struct PeriodicFixture {
  std::array<double, 9> box{kPeriodicEdge, 0.0, 0.0, 0.0, kPeriodicEdge,
                            0.0,           0.0, 0.0, kPeriodicEdge};
  std::array<double, 9> pos{1.0, 1.0, 1.1177,  1.0, 1.75545, 0.52884,
                            1.0, 0.24455, 0.52884};
  std::array<int, 3> z{8, 1, 1};
};

PeriodicResult eval_xtb(rgpot::XTBPot &pot, const std::array<double, 9> &pos,
                     const std::array<int, 3> &z,
                     const std::array<double, 9> &box) {
  std::array<double, 9> f{};
  rgpot::ForceOut out{f.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({3, pos.data(), z.data(), box.data()}, &out);
  return {out.energy, out.has_stress,
          {out.stress[0], out.stress[1], out.stress[2], out.stress[3],
           out.stress[4], out.stress[5], out.stress[6], out.stress[7],
           out.stress[8]}};
}

} // namespace

TEST_CASE("XTBPot treats a nonzero box as periodic", "[xtb][linked][periodic]") {
  PeriodicFixture fx;
  rgpot::XTBPot pot(periodicConfig());
  const auto base = eval_xtb(pot, fx.pos, fx.z, fx.box);

  auto imaged = fx.pos;
  imaged[6] += kPeriodicEdge; // second H, one lattice vector along x
  imaged[8] -= kPeriodicEdge; // and along -z
  const auto shifted = eval_xtb(pot, imaged, fx.z, fx.box);
  REQUIRE_THAT(shifted.energy, WithinAbs(base.energy, 1e-7));
}

TEST_CASE("XTBPot does not report a stress libxtb did not compute",
          "[xtb][linked][periodic][stress]") {
  PeriodicFixture fx;
  rgpot::XTBPot pot(periodicConfig());
  const auto periodic = eval_xtb(pot, fx.pos, fx.z, fx.box);
  REQUIRE(periodic.has_stress == 0);
}

TEST_CASE("XTBPot switches between periodic and isolated boxes",
          "[xtb][linked][periodic]") {
  PeriodicFixture fx;
  rgpot::XTBPot pot(periodicConfig());
  const std::array<double, 9> none{};
  const auto isolated_first = eval_xtb(pot, fx.pos, fx.z, none);
  const auto periodic = eval_xtb(pot, fx.pos, fx.z, fx.box);
  const auto isolated_again = eval_xtb(pot, fx.pos, fx.z, none);
  REQUIRE_THAT(isolated_again.energy, WithinAbs(isolated_first.energy, 1e-9));
  REQUIRE(std::abs(periodic.energy - isolated_first.energy) > 1e-6);
}
