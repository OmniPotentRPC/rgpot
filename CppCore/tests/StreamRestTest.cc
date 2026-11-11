// MIT License
// Copyright 2023--present rgpot developers
//
// Stored and streaming evaluations of the potentials that did not yet
// walk the shared neighbour layer. Energies and forces agree to
// round-off, and a stream keeps no pair index.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/ReducedPair/ReducedPair.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#if defined(RGPOT_HAS_FORTRAN_POTS)
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace {

void setThreads(const char *value) {
  setenv("OMP_NUM_THREADS", value, 1);
}

bool relClose(double a, double b, double rel = 1e-12) {
  const double scale = std::max(std::abs(a), std::abs(b));
  return std::abs(a - b) <= rel * std::max(1.0, scale);
}

std::vector<double> fcc(int n, double a) {
  const double b[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  std::vector<double> pos;
  for (int ix = 0; ix < n; ++ix) {
    for (int iy = 0; iy < n; ++iy) {
      for (int iz = 0; iz < n; ++iz) {
        for (const auto &p : b) {
          pos.push_back((ix + p[0]) * a);
          pos.push_back((iy + p[1]) * a);
          pos.push_back((iz + p[2]) * a);
        }
      }
    }
  }
  return pos;
}

void nudge(std::vector<double> &pos) {
  if (pos.size() < 3) {
    return;
  }
  pos[0] += 0.04;
  pos[pos.size() / 2] -= 0.03;
  pos[pos.size() - 1] += 0.02;
}

void requireMatch(const std::vector<double> &ga, double ea, const std::vector<double> &gb,
                  double eb) {
  REQUIRE(relClose(ea, eb));
  REQUIRE(ga.size() == gb.size());
  for (std::size_t i = 0; i < ga.size(); ++i) {
    REQUIRE(relClose(ga[i], gb[i]));
  }
}

} // namespace

TEST_CASE("reduced Lennard-Jones stream matches the stored list", "[nlist][stream][reduced]") {
  setThreads("1");
  auto pos = fcc(3, std::pow(2.0, 1.0 / 6.0) * std::sqrt(2.0));
  nudge(pos);
  const auto n = static_cast<int32_t>(pos.size() / 3);
  std::vector<double> gs(pos.size()), gw(pos.size());
  double es = 0;
  double ew = 0;
  rgpot_pair_set_neighbour_mode(1);
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 2.5, n, pos.data(), &es, gs.data()) == 0);
  const auto slots = rgpot_pair_neighbour_slots();
  REQUIRE(slots > 0);
  rgpot_pair_set_neighbour_mode(2);
  REQUIRE(rgpot_lj_gradient(1.0, 1.0, 2.5, n, pos.data(), &ew, gw.data()) == 0);
  REQUIRE(rgpot_pair_neighbour_slots() == 0);
  requireMatch(gs, es, gw, ew);

  double em = 0;
  std::vector<double> gm(pos.size());
  rgpot_pair_set_neighbour_mode(1);
  REQUIRE(rgpot_morse_gradient(1.0, 6.0, 1.0, 2.5, n, pos.data(), &em, gm.data()) == 0);
  REQUIRE(rgpot_pair_neighbour_slots() > 0);
  rgpot_pair_set_neighbour_mode(2);
  double emw = 0;
  std::vector<double> gmw(pos.size());
  REQUIRE(rgpot_morse_gradient(1.0, 6.0, 1.0, 2.5, n, pos.data(), &emw, gmw.data()) == 0);
  REQUIRE(rgpot_pair_neighbour_slots() == 0);
  requireMatch(gm, em, gmw, emw);
  rgpot_pair_set_neighbour_mode(1);
}

TEST_CASE("reduced binary and arithmetic streams match the stored list",
          "[nlist][stream][reduced]") {
  setThreads("1");
  auto pos = fcc(2, 1.2);
  nudge(pos);
  const auto n = static_cast<int32_t>(pos.size() / 3);
  std::vector<double> sigma(static_cast<std::size_t>(n), 1.0);
  sigma[0] = 1.05;
  sigma[3] = 0.95;
  std::vector<double> gs(pos.size()), gw(pos.size());
  double es = 0;
  double ew = 0;
  rgpot_pair_set_neighbour_mode(1);
  REQUIRE(rgpot_lj_arithmetic_gradient(sigma.data(), 2.2, n, pos.data(), &es, gs.data()) == 0);
  rgpot_pair_set_neighbour_mode(2);
  REQUIRE(rgpot_lj_arithmetic_gradient(sigma.data(), 2.2, n, pos.data(), &ew, gw.data()) == 0);
  requireMatch(gs, es, gw, ew);

  const double eps[3] = {1.0, 0.8, 1.2};
  const double sig[3] = {1.0, 0.9, 1.1};
  rgpot_pair_set_neighbour_mode(1);
  REQUIRE(rgpot_binary_lj_gradient(n / 2, eps, sig, 2.2, n, pos.data(), &es, gs.data()) == 0);
  rgpot_pair_set_neighbour_mode(2);
  REQUIRE(rgpot_binary_lj_gradient(n / 2, eps, sig, 2.2, n, pos.data(), &ew, gw.data()) == 0);
  requireMatch(gs, es, gw, ew);
  rgpot_pair_set_neighbour_mode(1);
}

TEST_CASE("EMT stream matches the stored list", "[nlist][stream][emt]") {
  setThreads("1");
  auto pos = fcc(4, 3.615);
  nudge(pos);
  const auto n = static_cast<long>(pos.size() / 3);
  rgpot::types::AtomMatrix xyz(n, 3);
  std::vector<int> z(static_cast<std::size_t>(n), 29);
  for (long i = 0; i < n; ++i) {
    xyz(i, 0) = pos[static_cast<std::size_t>(3 * i)];
    xyz(i, 1) = pos[static_cast<std::size_t>(3 * i + 1)];
    xyz(i, 2) = pos[static_cast<std::size_t>(3 * i + 2)];
  }
  const double side = 3 * 3.615;
  const std::array<std::array<double, 3>, 3> box{{{side, 0, 0}, {0, side, 0}, {0, 0, side}}};
  rgpot::EMTPot stored;
  rgpot::EMTPot stream;
  stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
  stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto a = stored(xyz, z, box);
  const auto b = stream(xyz, z, box);
  REQUIRE(stored.neighbourSlots() > 0);
  REQUIRE(stream.neighbourSlots() == 0);
  REQUIRE(relClose(std::get<0>(a), std::get<0>(b)));
  const auto &fa = std::get<1>(a);
  const auto &fb = std::get<1>(b);
  for (long i = 0; i < n; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE(relClose(fa(i, k), fb(i, k)));
    }
  }
}

#if defined(RGPOT_HAS_FORTRAN_POTS)
TEST_CASE("Fortran gathers stream on the shared layer", "[nlist][stream][fortran]") {
  setThreads("1");
  rgpot::types::AtomMatrix xyz{{7.28500, 7.28500, 7.28500},   {8.64250, 8.64250, 8.64250},
                               {10.00000, 10.00000, 7.28500}, {11.35750, 11.35750, 8.64250},
                               {10.00000, 7.28500, 10.00000}, {11.35750, 8.64250, 11.35750},
                               {7.28500, 10.00000, 10.00000}, {8.64250, 11.35750, 11.35750}};
  xyz(0, 0) += 0.05;
  xyz(3, 1) -= 0.04;
  const std::vector<int> si(8, 14);
  const std::array<std::array<double, 3>, 3> box{{{20, 0, 0}, {0, 20, 0}, {0, 0, 20}}};

  auto check = [&](auto &stored, auto &stream, const std::vector<int> &z) {
    stored.setNeighbourMode(rgpot::nlist::Mode::Stored);
    stream.setNeighbourMode(rgpot::nlist::Mode::Stream);
    const auto a = stored(xyz, z, box);
    const auto b = stream(xyz, z, box);
    REQUIRE(stored.neighbourSlots() > 0);
    REQUIRE(stream.neighbourSlots() == 0);
    REQUIRE(relClose(std::get<0>(a), std::get<0>(b)));
    const auto &fa = std::get<1>(a);
    const auto &fb = std::get<1>(b);
    for (std::size_t i = 0; i < fa.rows(); ++i) {
      for (std::size_t k = 0; k < 3; ++k) {
        REQUIRE(relClose(fa(i, k), fb(i, k)));
      }
    }
  };

  rgpot::fortranpots::SWPot sws;
  rgpot::fortranpots::SWPot sww;
  check(sws, sww, si);
  rgpot::fortranpots::TersoffPot tes;
  rgpot::fortranpots::TersoffPot tew;
  check(tes, tew, si);
  rgpot::fortranpots::EDIPPot eds;
  rgpot::fortranpots::EDIPPot edw;
  check(eds, edw, si);
  rgpot::fortranpots::LenoskyPot lns;
  rgpot::fortranpots::LenoskyPot lnw;
  check(lns, lnw, si);

  rgpot::types::AtomMatrix al{{7.975, 7.975, 7.975},
                              {10.0, 10.0, 7.975},
                              {10.0, 7.975, 10.0},
                              {7.975, 10.0, 10.0}};
  al(1, 2) += 0.03;
  const std::vector<int> zal(4, 13);
  rgpot::fortranpots::EAMAlPot eams;
  rgpot::fortranpots::EAMAlPot eamw;
  eams.setNeighbourMode(rgpot::nlist::Mode::Stored);
  eamw.setNeighbourMode(rgpot::nlist::Mode::Stream);
  const auto ea = eams(al, zal, box);
  const auto eb = eamw(al, zal, box);
  REQUIRE(eams.neighbourSlots() > 0);
  REQUIRE(eamw.neighbourSlots() == 0);
  REQUIRE(relClose(std::get<0>(ea), std::get<0>(eb)));
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t k = 0; k < 3; ++k) {
      REQUIRE(relClose(std::get<1>(ea)(i, k), std::get<1>(eb)(i, k)));
    }
  }
}
#endif
