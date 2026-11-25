// MIT License
// Copyright 2023--present rgpot developers
//
// ZBL checks against LAMMPS pair_style zbl, plus the switching, cutoff and
// species-table behaviour of the rgpot port.

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "TestFixtures.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::test_fixtures::cubicCell;
using rgpot::types::AtomMatrix;

namespace {

/// Si-Au dimer separated by (1.2, 1.3, 1.4) Angstrom, centred in a 20 A
/// cube: eOn's ZBL fixture (client/potentials/ZBL/wip_explorations
/// generates it, client/unit_tests/ZBLPotTest.cpp consumes it).
AtomMatrix siAuDimer() {
  return AtomMatrix{{9.40, 9.35, 9.30}, {10.60, 10.65, 10.70}};
}

} // namespace

TEST_CASE("ZBLPot against LAMMPS on a Si-Au dimer", "[ZBLPot]") {
  // LAMMPS pair_style zbl 2.0 2.5 in metal units; the same numbers eOn
  // pins in client/unit_tests/ZBLPotTest.cpp. Tolerances follow the
  // printed precision of the LAMMPS log and force dump.
  rgpot::ZBLPot pot;
  const auto positions = siAuDimer();
  const std::vector<int> atmtypes{14, 79};

  auto [energy, forces, variance] = pot(positions, atmtypes, cubicCell(20.0));
  (void)variance;

  REQUIRE_THAT(energy, WithinAbs(0.38537731, 1e-8));

  const std::array<std::array<double, 3>, 2> expected{
      {{-2.37926, -2.57753, -2.7758}, {2.37926, 2.57753, 2.7758}}};
  for (size_t i = 0; i < forces.rows(); ++i) {
    for (size_t j = 0; j < forces.cols(); ++j) {
      REQUIRE_THAT(forces(i, j), WithinAbs(expected[i][j], 1e-5));
    }
  }
}

TEST_CASE("ZBLPot switching zeroes energy and force at the cutoff",
          "[ZBLPot]") {
  rgpot::ZBLPot pot;
  const std::vector<int> atmtypes{14, 79};
  const auto cell = cubicCell(20.0);
  const double cut = pot.config().cut_global;

  const AtomMatrix atCut{{5.0, 5.0, 5.0}, {5.0 + cut, 5.0, 5.0}};
  auto [e_cut, f_cut, v_cut] = pot(atCut, atmtypes, cell);
  (void)v_cut;
  REQUIRE_THAT(e_cut, WithinAbs(0.0, 1e-12));
  for (size_t i = 0; i < f_cut.rows(); ++i) {
    for (size_t j = 0; j < f_cut.cols(); ++j) {
      REQUIRE_THAT(f_cut(i, j), WithinAbs(0.0, 1e-12));
    }
  }

  const AtomMatrix beyond{{5.0, 5.0, 5.0}, {5.0 + cut + 0.5, 5.0, 5.0}};
  auto [e_far, f_far, v_far] = pot(beyond, atmtypes, cell);
  (void)v_far;
  REQUIRE_THAT(e_far, WithinAbs(0.0, 1e-15));
  for (size_t i = 0; i < f_far.rows(); ++i) {
    for (size_t j = 0; j < f_far.cols(); ++j) {
      REQUIRE_THAT(f_far(i, j), WithinAbs(0.0, 1e-15));
    }
  }
}

TEST_CASE("ZBLPot rebuilds its tables when the species change", "[ZBLPot]") {
  // The coefficient tables are keyed on the set of atomic numbers, so one
  // instance reused across systems never carries stale screening lengths.
  rgpot::ZBLPot pot;
  const auto positions = siAuDimer();
  const auto cell = cubicCell(20.0);

  auto [e_siau, f_siau, v_siau] =
      pot(positions, std::vector<int>{14, 79}, cell);
  auto [e_auau, f_auau, v_auau] =
      pot(positions, std::vector<int>{79, 79}, cell);
  auto [e_sisi, f_sisi, v_sisi] =
      pot(positions, std::vector<int>{14, 14}, cell);
  auto [e_again, f_again, v_again] =
      pot(positions, std::vector<int>{14, 79}, cell);
  (void)v_siau;
  (void)v_auau;
  (void)v_sisi;
  (void)v_again;

  // References from an independent double-precision evaluation of this
  // kernel, cross-checked against LAMMPS for the Si-Au pair above.
  REQUIRE_THAT(e_siau, WithinRel(0.38537730883, 1e-9));
  REQUIRE_THAT(e_auau, WithinRel(0.961609764914, 1e-9));
  REQUIRE_THAT(e_sisi, WithinRel(0.167604056792, 1e-9));
  REQUIRE_THAT(e_again, WithinRel(e_siau, 1e-15));
}

TEST_CASE("ZBLPot periodic cell matches a replicated open supercell",
          "[ZBLPot][periodic][stress]") {
  const rgpot::ZBLPot pot(rgpot::ZBLConfig{.cut_inner = 1.8, .cut_global = 2.4});
  REQUIRE(pot.caps().periodic);
  REQUIRE(pot.caps().stress);

  const bool triclinic = GENERATE(false, true);
  const double a = 2.8, side = 2.0 * a;
  const double h[9] = {side, 0.0, 0.0,
                       triclinic ? 0.1 * side : 0.0, side, 0.0,
                       triclinic ? -0.08 * side : 0.0, triclinic ? 0.05 * side : 0.0, side};
  const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
  std::vector<double> pos;
  std::vector<int> z;
  for (int c = 0; c < 8; ++c) {
    for (const auto &s : basis) {
      const double f[3] = {((c & 1) + s[0]) / 2.0, (((c >> 1) & 1) + s[1]) / 2.0,
                           (((c >> 2) & 1) + s[2]) / 2.0};
      const int idx = static_cast<int>(z.size());
      for (int d = 0; d < 3; ++d) {
        const double r = f[0] * h[d] + f[1] * h[3 + d] + f[2] * h[6 + d];
        pos.push_back(r + 0.05 * a * std::sin(1.37 * idx + 0.9 * d));
      }
      z.push_back(idx % 2 == 0 ? 14 : 79);
    }
  }
  const std::size_t n = z.size();

  std::vector<double> f_pbc(3 * n, 0.0), e_pbc(n, 0.0), v_pbc(6 * n, 0.0);
  rgpot::ForceOut out_pbc{f_pbc.data(), 0.0, 0.0, {}, 0};
  pot.forcePerAtom({n, pos.data(), z.data(), h}, &out_pbc, {e_pbc.data(), v_pbc.data()});
  REQUIRE(out_pbc.energy > 0.0);
  REQUIRE(out_pbc.has_stress == 1);

  std::vector<double> super_pos(pos);
  std::vector<int> super_z(z);
  for (int i0 = -1; i0 <= 1; ++i0) {
    for (int i1 = -1; i1 <= 1; ++i1) {
      for (int i2 = -1; i2 <= 1; ++i2) {
        if (i0 == 0 && i1 == 0 && i2 == 0) {
          continue;
        }
        for (std::size_t k = 0; k < n; ++k) {
          for (int d = 0; d < 3; ++d) {
            super_pos.push_back(pos[3 * k + static_cast<std::size_t>(d)] +
                                i0 * h[d] + i1 * h[3 + d] + i2 * h[6 + d]);
          }
          super_z.push_back(z[k]);
        }
      }
    }
  }
  const std::size_t n_super = super_z.size();
  std::vector<double> f_open(3 * n_super, 0.0), e_open(n_super, 0.0), v_open(6 * n_super, 0.0);
  rgpot::ForceOut out_open{f_open.data(), 0.0, 0.0, {}, 0};
  pot.forcePerAtom({n_super, super_pos.data(), super_z.data(), nullptr}, &out_open,
                   {e_open.data(), v_open.data()});

  double central_e = 0.0;
  double central_v[6] = {};
  for (std::size_t k = 0; k < n; ++k) {
    central_e += e_open[k];
    REQUIRE_THAT(e_pbc[k], WithinRel(e_open[k], 1e-12));
    for (std::size_t d = 0; d < 3; ++d) {
      REQUIRE_THAT(f_pbc[3 * k + d], WithinAbs(f_open[3 * k + d], 1e-11));
    }
    for (std::size_t c = 0; c < 6; ++c) {
      central_v[c] += v_open[6 * k + c];
      REQUIRE_THAT(v_pbc[6 * k + c], WithinAbs(v_open[6 * k + c], 1e-11));
    }
  }
  REQUIRE_THAT(out_pbc.energy, WithinRel(central_e, 1e-12));
  const double volume = rgpot::cellVolume(h);
  const int rows[6] = {0, 1, 2, 1, 0, 0};
  const int cols[6] = {0, 1, 2, 2, 2, 1};
  for (int c = 0; c < 6; ++c) {
    REQUIRE_THAT(-volume * out_pbc.stress[3 * rows[c] + cols[c]],
                 WithinAbs(central_v[c], 1e-10));
  }

  constexpr double h_fd = 1e-6;
  auto evalE = [&](const std::vector<double> &p, const double *cell) {
    std::vector<double> f_tmp(3 * n, 0.0);
    rgpot::ForceOut sample{f_tmp.data(), 0.0, 0.0, {}, 0};
    pot.forceImpl({n, p.data(), z.data(), cell}, &sample);
    return sample.energy;
  };
  for (int d = 0; d < 3; ++d) {
    auto plus = pos, minus = pos;
    plus[static_cast<std::size_t>(d)] += h_fd;
    minus[static_cast<std::size_t>(d)] -= h_fd;
    const double fd_f = -(evalE(plus, h) - evalE(minus, h)) / (2.0 * h_fd);
    REQUIRE_THAT(f_pbc[static_cast<std::size_t>(d)], WithinAbs(fd_f, 1e-6));
  }
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      auto strained = [&](double eps) {
        auto moved = pos;
        double cell[9];
        std::copy(h, h + 9, cell);
        for (std::size_t a_i = 0; a_i < n; ++a_i) {
          moved[3 * a_i + static_cast<std::size_t>(col)] +=
              eps * pos[3 * a_i + static_cast<std::size_t>(row)];
        }
        for (int ax = 0; ax < 3; ++ax) {
          cell[3 * ax + col] += eps * h[3 * ax + row];
        }
        return evalE(moved, cell);
      };
      const double fd_s = (strained(h_fd) - strained(-h_fd)) / (2.0 * h_fd * volume);
      REQUIRE_THAT(out_pbc.stress[3 * row + col], WithinAbs(fd_s, 1e-6));
    }
  }
}

TEST_CASE("ZBLPot rejects an invalid cutoff pair", "[ZBLPot]") {
  // Named configs: a brace list with a comma cannot travel through a Catch
  // macro argument.
  const rgpot::ZBLConfig inverted{.cut_inner = 2.5, .cut_global = 2.0};
  const rgpot::ZBLConfig coincident{.cut_inner = 2.0, .cut_global = 2.0};
  const rgpot::ZBLConfig zeroInner{.cut_inner = 0.0, .cut_global = 2.5};
  const rgpot::ZBLConfig wide{.cut_inner = 2.0, .cut_global = 4.5};

  REQUIRE_THROWS_AS(rgpot::ZBLPot{inverted}, std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::ZBLPot{coincident}, std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::ZBLPot{zeroInner}, std::invalid_argument);
  REQUIRE_NOTHROW(rgpot::ZBLPot{wide});
}

TEST_CASE("ZBLPot parameter fingerprint tracks the config", "[ZBLPot]") {
  const rgpot::ZBLPot defaults;
  const rgpot::ZBLPot wider{rgpot::ZBLConfig{.cut_global = 4.5}};
  const rgpot::ZBLPot sameAsDefaults{rgpot::ZBLConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != wider.paramsKey());
  REQUIRE(defaults.getType() == rgpot::PotType::ZBL);
  REQUIRE(defaults.config().cut_inner == 2.0);
  REQUIRE(defaults.config().cut_global == 2.5);
}

TEST_CASE("ZBLPot declares kCellOptional and an open call is the cluster result",
          "[ZBLPot][periodic][cell-optional]") {
  const rgpot::ZBLPot pot;
  // A kernel that supports a periodic cell and evaluates an open system
  // without one: a call with no cell is not an error.
  STATIC_REQUIRE(rgpot::kernelCellOptional<rgpot::ZBLPot>());
  CHECK(pot.caps().periodic);
  CHECK_FALSE(rgpot::kernelNeedsCell(pot));

  const std::vector<double> pos{0.0, 0.0, 0.0, 1.5, 0.4, -0.3};
  const std::vector<int> z{14, 79};
  const double zero_box[9] = {};
  // A cell far wider than twice the cutoff changes nothing for a close pair.
  const double wide_box[9] = {100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0};
  // A box extent used as a periodic cell would wrap the pair through its face.
  const double tight_box[9] = {2.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};

  std::vector<double> f_null(6, 0.0), f_zero(6, 0.0), f_wide(6, 0.0), f_tight(6, 0.0);
  rgpot::ForceOut out_null{f_null.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut out_zero{f_zero.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut out_wide{f_wide.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut out_tight{f_tight.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({2, pos.data(), z.data(), nullptr}, &out_null);
  pot.forceImpl({2, pos.data(), z.data(), zero_box}, &out_zero);
  pot.forceImpl({2, pos.data(), z.data(), wide_box}, &out_wide);
  pot.forceImpl({2, pos.data(), z.data(), tight_box}, &out_tight);

  REQUIRE(out_null.energy > 1.0);
  CHECK(out_zero.energy == out_null.energy);
  CHECK_THAT(out_wide.energy, WithinRel(out_null.energy, 1e-14));
  CHECK(out_null.has_stress == 0);
  CHECK(out_zero.has_stress == 0);
  CHECK(out_wide.has_stress == 1);
  for (std::size_t k = 0; k < 6; ++k) {
    CHECK(f_zero[k] == f_null[k]);
    CHECK_THAT(f_wide[k], WithinAbs(f_null[k], 1e-11));
  }
  CHECK(std::abs(out_tight.energy - out_null.energy) > 1.0);
}

TEST_CASE("ZBLPot folds pairs on a rotated cell and rejects malformed cells",
          "[ZBLPot][periodic][stress]") {
  const rgpot::ZBLPot pot;
  const double side = 5.0;
  const double std_box[9] = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  // 90-degree rotation in the xy plane: a = (0, L, 0), b = (-L, 0, 0),
  // c = (0, 0, L), so the diagonal entries in x and y are zero while
  // det(H) = L^3 > 0.
  const double rot_box[9] = {0.0, side, 0.0, -side, 0.0, 0.0, 0.0, 0.0, side};

  // Si-Si dimer separated by 0.2 A across the first periodic boundary:
  // unwrapped separation is 4.8 A > cut_global (2.5 A), so dropping
  // periodicity gives 0 eV while the folded image gives ~4053.35 eV.
  const std::vector<double> pos_std{0.1, 0.0, 0.0, 4.9, 0.0, 0.0};
  const std::vector<double> pos_rot{0.0, 0.1, 0.0, 0.0, 4.9, 0.0};
  const std::vector<int> z{14, 14};

  std::vector<double> f_std(6, 0.0), f_rot(6, 0.0), e_std(2, 0.0), e_rot(2, 0.0),
      v_std(12, 0.0), v_rot(12, 0.0);
  rgpot::ForceOut out_std{f_std.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut out_rot{f_rot.data(), 0.0, 0.0, {}, 0};
  pot.forcePerAtom({2, pos_std.data(), z.data(), std_box}, &out_std,
                   {e_std.data(), v_std.data()});
  pot.forcePerAtom({2, pos_rot.data(), z.data(), rot_box}, &out_rot,
                   {e_rot.data(), v_rot.data()});

  CHECK_THAT(out_std.energy, WithinRel(4053.35, 1e-4));
  CHECK_THAT(out_rot.energy, WithinRel(out_std.energy, 1e-14));
  CHECK(out_std.has_stress == 1);
  CHECK(out_rot.has_stress == 1);
  for (std::size_t i = 0; i < 2; ++i) {
    CHECK_THAT(e_rot[i], WithinRel(e_std[i], 1e-14));
  }
  // Forces and stress rotate by (x, y, z) -> (-y, x, z).
  CHECK_THAT(f_rot[0], WithinAbs(-f_std[1], 1e-12));
  CHECK_THAT(f_rot[1], WithinAbs(f_std[0], 1e-12));
  CHECK_THAT(f_rot[2], WithinAbs(f_std[2], 1e-12));
  CHECK_THAT(out_rot.stress[4], WithinRel(out_std.stress[0], 1e-14));
  CHECK_THAT(out_rot.stress[0], WithinRel(out_std.stress[4], 1e-14));
  CHECK_THAT(out_rot.stress[8], WithinRel(out_std.stress[8], 1e-14));

  // Null and all-zero cells use free boundaries.
  const double zero_box[9] = {};
  std::vector<double> f_null(6, 0.0), f_zero(6, 0.0);
  rgpot::ForceOut out_null{f_null.data(), 0.0, 0.0, {}, 0};
  rgpot::ForceOut out_zero{f_zero.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl({2, pos_std.data(), z.data(), nullptr}, &out_null);
  pot.forceImpl({2, pos_std.data(), z.data(), zero_box}, &out_zero);
  CHECK(out_null.energy == 0.0);
  CHECK(out_zero.energy == 0.0);
  CHECK(out_null.has_stress == 0);
  CHECK(out_zero.has_stress == 0);

  // Malformed non-zero cells (non-positive diagonal or singular sheared cell)
  // throw std::invalid_argument instead of silently falling back to free_box.
  const double neg_diag[9] = {-side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, side};
  const double zero_diag[9] = {side, 0.0, 0.0, 0.0, side, 0.0, 0.0, 0.0, 0.0};
  const double singular_shear[9] = {side, side, 0.0, side, side, 0.0, 0.0, 0.0, side};
  rgpot::ForceOut out_bad{f_std.data(), 0.0, 0.0, {}, 0};
  CHECK_THROWS_AS(pot.forceImpl({2, pos_std.data(), z.data(), neg_diag}, &out_bad),
                  std::invalid_argument);
  CHECK_THROWS_AS(pot.forceImpl({2, pos_std.data(), z.data(), zero_diag}, &out_bad),
                  std::invalid_argument);
  CHECK_THROWS_AS(pot.forceImpl({2, pos_std.data(), z.data(), singular_shear}, &out_bad),
                  std::invalid_argument);
}
