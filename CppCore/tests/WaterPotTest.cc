// MIT License
// Copyright 2023--present rgpot developers

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/Water/WaterPots.hpp"
#include "rgpot/Water/potential_base.hpp"

using Catch::Matchers::WithinAbs;

namespace {

// One flexible water, off the TIP4P equilibrium geometry, in a cell
// larger than the 8.5 angstrom cutoff. Order is H, H, O.
rgpot::types::AtomMatrix waterPositions() {
  rgpot::types::AtomMatrix positions(3, 3);
  const double xyz[3][3] = {
      {0.80, 0.10, 0.50},
      {-0.70, 0.05, 0.55},
      {0.0, 0.0, 0.0},
  };
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[i][k];
    }
  }
  return positions;
}

const std::array<std::array<double, 3>, 3> kBox = {{
    {40.0, 0.0, 0.0},
    {0.0, 40.0, 0.0},
    {0.0, 0.0, 40.0},
}};

double maxForce(const rgpot::types::AtomMatrix &forces) {
  double worst = 0.0;
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) + forces(i, 1) * forces(i, 1) +
                      forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

void requireNetForce(const rgpot::types::AtomMatrix &forces) {
  double sum[3] = {0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    sum[0] += forces(i, 0);
    sum[1] += forces(i, 1);
    sum[2] += forces(i, 2);
  }
  REQUIRE_THAT(sum[0], WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(sum[1], WithinAbs(0.0, 1e-8));
  REQUIRE_THAT(sum[2], WithinAbs(0.0, 1e-8));
}

// `copies` of `pos` along each axis of the diagonal `box`. Rows keep the
// H-pairs, then O, then Pt order, so the first copy of every block leads.
rgpot::types::AtomMatrix repeatWater(const rgpot::types::AtomMatrix &pos,
                                     const std::vector<int> &types,
                                     const std::array<double, 3> &box,
                                     const std::array<int, 3> &copies,
                                     std::vector<int> &bigTypes) {
  const int total = copies[0] * copies[1] * copies[2];
  rgpot::types::AtomMatrix big(pos.rows() * static_cast<std::size_t>(total), 3);
  bigTypes.clear();
  std::size_t row = 0;
  std::size_t start = 0;
  while (start < pos.rows()) {
    std::size_t end = start;
    while (end < pos.rows() && types[end] == types[start]) {
      ++end;
    }
    for (int a = 0; a < copies[0]; ++a) {
      for (int b = 0; b < copies[1]; ++b) {
        for (int c = 0; c < copies[2]; ++c) {
          const double shift[3] = {a * box[0], b * box[1], c * box[2]};
          for (std::size_t i = start; i < end; ++i, ++row) {
            for (int k = 0; k < 3; ++k) {
              big(row, k) = pos(i, k) + shift[k];
            }
            bigTypes.push_back(types[i]);
          }
        }
      }
    }
    start = end;
  }
  return big;
}

template <class Pot>
void requireThinBoxMatchesRepeat(Pot &pot, const rgpot::types::AtomMatrix &pos,
                                 const std::vector<int> &types,
                                 const std::array<double, 3> &box,
                                 const std::array<int, 3> &copies) {
  const auto diag = [](const std::array<double, 3> &l) {
    return std::array<std::array<double, 3>, 3>{
        {{l[0], 0.0, 0.0}, {0.0, l[1], 0.0}, {0.0, 0.0, l[2]}}};
  };
  std::vector<int> bigTypes;
  const auto big = repeatWater(pos, types, box, copies, bigTypes);
  const std::array<double, 3> bigBox = {copies[0] * box[0], copies[1] * box[1],
                                        copies[2] * box[2]};
  for (int k = 0; k < 3; ++k) {
    REQUIRE(bigBox[static_cast<std::size_t>(k)] >= 17.0);
  }
  const auto [e1, f1, v1] = pot(pos, types, diag(box));
  const auto [e2, f2, v2] = pot(big, bigTypes, diag(bigBox));
  const int total = copies[0] * copies[1] * copies[2];
  REQUIRE_THAT(e1, WithinAbs(e2 / total, 1e-11));
  // Row i of the small cell is row (block start) * total + offset in big.
  std::size_t start = 0;
  while (start < pos.rows()) {
    std::size_t end = start;
    while (end < pos.rows() && types[end] == types[start]) {
      ++end;
    }
    for (std::size_t i = start; i < end; ++i) {
      for (int k = 0; k < 3; ++k) {
        REQUIRE_THAT(f1(i, k), WithinAbs(f2(start * total + (i - start), k), 1e-11));
      }
    }
    start = end;
  }
  (void)v1;
  (void)v2;
}

class PtLjProbe : public forcefields::PotentialBase {
public:
  void eval(int nAtoms, double positions[], double forces[], double &energy,
            double const periods[], bool const fixed[]) {
    computePt(nAtoms, positions, forces, energy, periods, fixed);
  }
};

} // namespace

TEST_CASE("TIP4P energy and force on one distorted water", "[water]") {
  rgpot::TIP4PPot pot(rgpot::WaterConfig{});
  const std::vector<int> types = {1, 1, 8};
  const auto [energy, forces, variance] = pot(waterPositions(), types, kBox);
  std::printf("TIP4P_ENERGY %.12f TIP4P_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(0.141344038570, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(4.479603395144, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.config().cutoff == 8.5);
  REQUIRE(pot.config().switching_width == 1.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.get_type() == rgpot::PotType::TIP4P);
  requireNetForce(forces);
  const auto [again, forces2, variance2] = pot(waterPositions(), types, kBox);
  REQUIRE_THAT(again, WithinAbs(energy, 0.0));
  (void)forces2;
  (void)variance2;
}

TEST_CASE("SPC/E energy and force on one distorted water", "[water]") {
  rgpot::SPCEPot pot(rgpot::WaterConfig{});
  const std::vector<int> types = {1, 1, 8};
  const auto [energy, forces, variance] = pot(waterPositions(), types, kBox);
  std::printf("SPCE_ENERGY %.12f SPCE_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(0.532963694950, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(8.458875831491, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.get_type() == rgpot::PotType::SPCE);
  requireNetForce(forces);
  rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  const auto [tipEnergy, tipForces, tipVar] = tip(waterPositions(), types, kBox);
  REQUIRE(energy != tipEnergy);
  (void)tipForces;
  (void)tipVar;
}

TEST_CASE("TIP4P-Pt energy on one water and one platinum", "[water]") {
  // The mirror plane is z = 0, the plane of the top platinum layer.
  // The water sits above that plane.
  rgpot::types::AtomMatrix positions(4, 3);
  positions(0, 0) = 0.80;
  positions(0, 1) = 0.10;
  positions(0, 2) = 2.50;
  positions(1, 0) = -0.70;
  positions(1, 1) = 0.05;
  positions(1, 2) = 2.55;
  positions(2, 0) = 0.0;
  positions(2, 1) = 0.0;
  positions(2, 2) = 2.0;
  positions(3, 0) = 0.0;
  positions(3, 1) = 0.0;
  positions(3, 2) = 0.0;
  const std::vector<int> types = {1, 1, 8, 78};
  rgpot::TIP4PPtPot pot(rgpot::WaterConfig{});
  const auto [energy, forces, variance] = pot(positions, types, kBox);
  std::printf("TIP4P_PT_ENERGY %.12f TIP4P_PT_MAX_FORCE %.12f\n", energy,
              maxForce(forces));
  REQUIRE_THAT(energy, WithinAbs(1.642427227258, 1e-9));
  REQUIRE_THAT(maxForce(forces), WithinAbs(15.396689248363, 1e-6));
  REQUIRE(variance == 0.0);
  REQUIRE(pot.paramsKey() != 0);
  REQUIRE(pot.get_type() == rgpot::PotType::TIP4PPt);
  for (std::size_t i = 0; i < forces.rows(); ++i) {
    REQUIRE(std::isfinite(forces(i, 0)));
    REQUIRE(std::isfinite(forces(i, 1)));
    REQUIRE(std::isfinite(forces(i, 2)));
  }
}

TEST_CASE("Water in a box thinner than twice the cutoff sums every image",
          "[water][images]") {
  const std::array<double, 3> box = {5.2, 6.1, 7.3};
  rgpot::types::AtomMatrix two(6, 3);
  const double xyz[6][3] = {{1.8, 1.1, 1.5}, {0.3, 1.05, 1.55}, {4.3, 3.6, 4.9},
                            {3.1, 4.6, 4.9}, {1.0, 1.0, 1.0},   {3.6, 3.9, 4.4}};
  for (int i = 0; i < 6; ++i) {
    for (int k = 0; k < 3; ++k) {
      two(i, k) = xyz[i][k];
    }
  }
  const std::vector<int> twoTypes = {1, 1, 1, 1, 8, 8};
  rgpot::TIP4PPot tip(rgpot::WaterConfig{});
  rgpot::SPCEPot spc(rgpot::WaterConfig{});
  requireThinBoxMatchesRepeat(tip, two, twoTypes, box, {4, 3, 3});
  requireThinBoxMatchesRepeat(spc, two, twoTypes, box, {4, 3, 3});
  // One molecule meets only its own images.
  const std::vector<int> oneTypes = {1, 1, 8};
  requireThinBoxMatchesRepeat(tip, waterPositions(), oneTypes, box, {4, 3, 3});
  requireThinBoxMatchesRepeat(spc, waterPositions(), oneTypes, box, {4, 3, 3});

  // TIP4P-Pt above a 2x2 Pt(100) layer, thin across the surface.
  const double a = 2.77;
  rgpot::types::AtomMatrix slab(7, 3);
  const double sxyz[7][3] = {{2.0, 1.1, 3.1}, {0.5, 1.05, 3.15}, {1.2, 1.0, 2.6},
                             {0.0, 0.0, 0.0}, {a, 0.0, 0.0},     {0.0, a, 0.0},
                             {a, a, 0.0}};
  for (int i = 0; i < 7; ++i) {
    for (int k = 0; k < 3; ++k) {
      slab(i, k) = sxyz[i][k];
    }
  }
  rgpot::TIP4PPtPot zp(rgpot::WaterConfig{});
  requireThinBoxMatchesRepeat(zp, slab, {1, 1, 8, 78, 78, 78, 78}, {2 * a, 2 * a, 30.0},
                              {4, 4, 1});
}

TEST_CASE("Water kernels refuse a sheared cell", "[water]") {
  rgpot::TIP4PPot pot(rgpot::WaterConfig{});
  const std::vector<int> types = {1, 1, 8};
  const std::array<std::array<double, 3>, 3> sheared = {{
      {40.0, 0.0, 0.0},
      {3.0, 40.0, 0.0},
      {0.0, 0.0, 40.0},
  }};
  REQUIRE_THROWS_AS(pot(waterPositions(), types, sheared), std::invalid_argument);
}

TEST_CASE("computePt sums the images of one atom in a thin cell", "[water][pt]") {
  PtLjProbe pot;
  double one[3] = {0.0, 0.0, 0.0};
  double small[3] = {2.77, 2.77, 2.77};
  double f1[3] = {};
  double e1 = 0.0;
  bool fixed1[1] = {false};
  pot.eval(1, one, f1, e1, small, fixed1);
  std::vector<double> big;
  for (int x = 0; x < 7; ++x) {
    for (int y = 0; y < 7; ++y) {
      for (int z = 0; z < 7; ++z) {
        big.push_back(x * 2.77);
        big.push_back(y * 2.77);
        big.push_back(z * 2.77);
      }
    }
  }
  double large[3] = {7 * 2.77, 7 * 2.77, 7 * 2.77};
  std::vector<double> f2(big.size(), 0.0);
  std::vector<char> fixed2(big.size() / 3, 0);
  double e2 = 0.0;
  pot.eval(343, big.data(), f2.data(), e2, large, reinterpret_cast<bool *>(fixed2.data()));
  REQUIRE(e1 < 0.0);
  REQUIRE_THAT(e1, WithinAbs(e2 / 343.0, 1e-10));
}

TEST_CASE("computePt keeps an LJ pair when one atom is fixed", "[water][pt]") {
  auto evalPair = [](bool fixed0, bool fixed1) {
    PtLjProbe pot;
    double positions[6] = {0.0, 0.0, 0.0, 3.0, 0.0, 0.0};
    double periods[3] = {40.0, 40.0, 40.0};
    bool fixed[2] = {fixed0, fixed1};
    std::array<double, 6> forces{};
    double energy = 0.0;
    pot.eval(2, positions, forces.data(), energy, periods, fixed);
    return std::pair<double, std::array<double, 6>>{energy, forces};
  };
  const auto free = evalPair(false, false);
  REQUIRE(free.first != 0.0);
  REQUIRE(free.second[0] != 0.0);
  REQUIRE(free.second[0] == -free.second[3]);
  REQUIRE(free.second[1] == 0.0);
  REQUIRE(free.second[2] == 0.0);

  const auto fix0 = evalPair(true, false);
  const auto fix1 = evalPair(false, true);
  REQUIRE(fix0.first == free.first);
  REQUIRE(fix1.first == free.first);
  REQUIRE(fix0.second == free.second);
  REQUIRE(fix1.second == free.second);

  const auto both = evalPair(true, true);
  REQUIRE(both.first == 0.0);
  for (double component : both.second) {
    REQUIRE(component == 0.0);
  }
}
