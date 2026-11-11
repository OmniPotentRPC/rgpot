// MIT License
// Copyright 2023--present rgpot developers
//
// A pair that starts between the cutoff and the skin, then moves inside
// the cutoff by less than half the skin, has to contribute. A strict
// list reused from the first call drops that pair and the energy stays
// zero. This fails on that cache and passes once the candidates reach
// the skin.

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace {

void setThreads(const char *value) {
#if defined(_WIN32)
  _putenv_s("OMP_NUM_THREADS", value == nullptr ? "" : value);
#else
  if (value == nullptr || value[0] == '\0') {
    unsetenv("OMP_NUM_THREADS");
  } else {
    setenv("OMP_NUM_THREADS", value, 1);
  }
#endif
}

bool relClose(double a, double b) {
  const double scale = std::max(std::abs(a), std::abs(b));
  return std::abs(a - b) <= 1e-12 * std::max(1.0, scale);
}

} // namespace

TEST_CASE("stored LJ keeps a pair that crosses the cutoff inside the skin",
          "[nlist][skin]") {
  setThreads("");
  // Default skin is 1 A. The pair starts past the cutoff and inside the
  // skin, then moves inside the cutoff by less than half the skin.
  rgpot::LJConfig cfg;
  cfg.cutoff = 3.0;
  cfg.psi = 1.0;
  cfg.u0 = 1.0;
  rgpot::LJPot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 5.0;
  xyz(0, 1) = 5.0;
  xyz(0, 2) = 5.0;
  xyz(1, 0) = 8.2;
  xyz(1, 1) = 5.0;
  xyz(1, 2) = 5.0;
  const std::vector<int> z{18, 18};
  const std::array<std::array<double, 3>, 3> box{{
      {{20.0, 0.0, 0.0}},
      {{0.0, 20.0, 0.0}},
      {{0.0, 0.0, 20.0}},
  }};
  const auto far = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(far)) < 1e-12);

  xyz(1, 0) = 7.9;
  const auto near = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(near)) > 1e-6);

  rgpot::LJPot fresh(cfg);
  fresh.setNeighbourMode(rgpot::nlist::Mode::Stored);
  const auto again = fresh(xyz, z, box);
  REQUIRE(relClose(std::get<0>(near), std::get<0>(again)));
  REQUIRE(relClose(std::get<1>(near)(0, 0), std::get<1>(again)(0, 0)));
}

TEST_CASE("stored Morse keeps a pair that crosses the cutoff inside the skin",
          "[nlist][skin]") {
  setThreads("");
  rgpot::MorseConfig cfg;
  cfg.cutoff = 5.0;
  rgpot::MorsePot pot(cfg);
  pot.setNeighbourMode(rgpot::nlist::Mode::Stored);
  // One fifth of the cutoff, which is also the PairSpec default of 1 A.
  const double skin = 1.0;
  rgpot::types::AtomMatrix xyz(2, 3);
  xyz(0, 0) = 2.0;
  xyz(0, 1) = 2.0;
  xyz(0, 2) = 2.0;
  xyz(1, 0) = 2.0 + cfg.cutoff + 0.2 * skin;
  xyz(1, 1) = 2.0;
  xyz(1, 2) = 2.0;
  const std::vector<int> z{78, 78};
  const std::array<std::array<double, 3>, 3> box{{
      {{30.0, 0.0, 0.0}},
      {{0.0, 30.0, 0.0}},
      {{0.0, 0.0, 30.0}},
  }};
  const auto far = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(far)) < 1e-12);

  xyz(1, 0) = 2.0 + cfg.cutoff - 0.2 * skin;
  const auto near = pot(xyz, z, box);
  REQUIRE(std::abs(std::get<0>(near)) > 1e-8);
  rgpot::MorsePot fresh(cfg);
  fresh.setNeighbourMode(rgpot::nlist::Mode::Stored);
  REQUIRE(relClose(std::get<0>(near), std::get<0>(fresh(xyz, z, box))));
}
