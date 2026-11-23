// MIT License
// Copyright 2023--present rgpot developers

// The two and three argument forms of the input guards are separate exported
// symbols. A binary built against the old two argument signature resolves the
// two argument symbol, so each one must keep matching the three argument form
// with the species check on.

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>

#include "rgpot/PotHelpers.hpp"

namespace {

struct Buffers {
  double pos[3] = {0.0, 0.0, 0.0};
  int species = 1;
  double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  double forces[3] = {0.0, 0.0, 0.0};
};

std::string thrown(auto &&call) {
  try {
    call();
  } catch (const std::invalid_argument &ex) {
    return ex.what();
  }
  return "";
}

} // namespace

TEST_CASE("rejectInputCall two arguments match three", "[pot-helpers-compat]") {
  Buffers b;
  const rgpot::ForceInput ok{1, b.pos, &b.species, b.box};
  const rgpot::ForceInput noSpecies{1, b.pos, nullptr, b.box};
  const rgpot::ForceInput noBox{1, b.pos, &b.species, nullptr};
  const rgpot::ForceInput empty{0, b.pos, &b.species, b.box};

  for (const bool needsCell : {true, false}) {
    for (const auto *in : {&ok, &noSpecies, &noBox, &empty}) {
      const int three = rgpot::rejectInputCall(*in, needsCell, true);
      const std::string threeMsg = rgpot::forceLastError();
      rgpot::setForceLastError("sentinel");
      const int two = rgpot::rejectInputCall(*in, needsCell);
      REQUIRE(two == three);
      if (three != 0) {
        REQUIRE(std::string(rgpot::forceLastError()) == threeMsg);
      } else {
        REQUIRE(std::string(rgpot::forceLastError()) == "sentinel");
      }
    }
  }
  REQUIRE(rgpot::rejectInputCall(noSpecies, false) ==
          rgpot::kInvalidParameter);
  REQUIRE(std::string(rgpot::forceLastError()) ==
          "invalid parameter: atomic numbers are null");
  REQUIRE(rgpot::rejectInputCall(noSpecies, false, false) == 0);
}

TEST_CASE("guardInputCall two arguments match three", "[pot-helpers-compat]") {
  Buffers b;
  const rgpot::ForceInput ok{1, b.pos, &b.species, b.box};
  const rgpot::ForceInput noSpecies{1, b.pos, nullptr, b.box};
  const rgpot::ForceInput noBox{1, b.pos, &b.species, nullptr};

  REQUIRE_NOTHROW(rgpot::guardInputCall(ok, true));
  REQUIRE_NOTHROW(rgpot::guardInputCall(ok, true, true));
  for (const auto *in : {&noSpecies, &noBox}) {
    const std::string two = thrown([&] { rgpot::guardInputCall(*in, true); });
    const std::string three =
        thrown([&] { rgpot::guardInputCall(*in, true, true); });
    REQUIRE_FALSE(two.empty());
    REQUIRE(two == three);
  }
  REQUIRE_NOTHROW(rgpot::guardInputCall(noSpecies, false, false));
  REQUIRE_THROWS_AS(rgpot::guardInputCall(noSpecies, false),
                    std::invalid_argument);
}

TEST_CASE("rejectForceCall three arguments match four", "[pot-helpers-compat]") {
  Buffers b;
  const rgpot::ForceInput ok{1, b.pos, &b.species, b.box};
  const rgpot::ForceInput noSpecies{1, b.pos, nullptr, b.box};
  const rgpot::ForceInput noBox{1, b.pos, &b.species, nullptr};
  const rgpot::ForceInput empty{0, b.pos, &b.species, b.box};
  rgpot::ForceOut withForces{};
  withForces.F = b.forces;
  rgpot::ForceOut noForces{};

  for (const bool needsCell : {true, false}) {
    for (const auto *in : {&ok, &noSpecies, &noBox, &empty}) {
      for (const rgpot::ForceOut *out : {&withForces, &noForces,
                                         static_cast<rgpot::ForceOut *>(
                                             nullptr)}) {
        const int four = rgpot::rejectForceCall(*in, out, needsCell, true);
        const std::string fourMsg = rgpot::forceLastError();
        rgpot::setForceLastError("sentinel");
        const int three = rgpot::rejectForceCall(*in, out, needsCell);
        REQUIRE(three == four);
        if (four != 0) {
          REQUIRE(std::string(rgpot::forceLastError()) == fourMsg);
        } else {
          REQUIRE(std::string(rgpot::forceLastError()) == "sentinel");
        }
      }
    }
  }
  REQUIRE(rgpot::rejectForceCall(noSpecies, &withForces, false) ==
          rgpot::kInvalidParameter);
  REQUIRE(rgpot::rejectForceCall(noSpecies, &withForces, false, false) == 0);
}

TEST_CASE("guardForceCall three arguments match four", "[pot-helpers-compat]") {
  Buffers b;
  const rgpot::ForceInput ok{1, b.pos, &b.species, b.box};
  const rgpot::ForceInput noSpecies{1, b.pos, nullptr, b.box};
  rgpot::ForceOut out{};
  out.F = b.forces;

  REQUIRE_NOTHROW(rgpot::guardForceCall(ok, &out, true));
  REQUIRE_NOTHROW(rgpot::guardForceCall(ok, &out, true, true));
  const std::string three =
      thrown([&] { rgpot::guardForceCall(noSpecies, &out, true); });
  const std::string four =
      thrown([&] { rgpot::guardForceCall(noSpecies, &out, true, true); });
  REQUIRE_FALSE(three.empty());
  REQUIRE(three == four);
  REQUIRE_NOTHROW(rgpot::guardForceCall(noSpecies, &out, false, false));
}
