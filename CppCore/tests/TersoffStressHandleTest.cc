// MIT License
// Copyright 2023--present rgpot developers
//
// TersoffPot behind PotentialHandle::from_impl: a stress request without
// atomic numbers comes back as RGPOT_INVALID_PARAMETER with a message that
// names them, and the same handle still answers a request that has them.

#include <array>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/fortran/FortranPots.hpp"
#include "rgpot/potential.hpp"

using rgpot::PotentialHandle;

TEST_CASE("a Tersoff stress request without species is an invalid parameter",
          "[PotentialHandle][stress][tersoff]") {
  rgpot::fortranpots::TersoffPot pot;
  auto handle = PotentialHandle::from_impl(pot);
  REQUIRE(handle.supports_stress());

  std::vector<double> pos{0.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  std::array<double, 9> box{20.0, 0, 0, 0, 20.0, 0, 0, 0, 20.0};

  rgpot::InputSpec bare(2, pos.data(), nullptr, box.data());
  bare.with_stress();
  try {
    (void)handle.calculate(bare);
    FAIL("the request was served");
  } catch (const rgpot::Error &e) {
    CHECK(e.status() == RGPOT_INVALID_PARAMETER);
    CHECK(std::string(e.what()).find("needs atomic numbers") !=
          std::string::npos);
  }

  std::vector<int> si{14, 14};
  rgpot::InputSpec named(pos, si, box.data());
  named.with_stress();
  const auto result = handle.calculate(named);
  CHECK(result.has_stress());
}
