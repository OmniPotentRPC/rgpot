// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>

#include "rgpot/CalculatorGroup.hpp"

// The abort flag is process state read by the exit handler that
// finalizeMpiAtExit registers. This binary never calls
// finalizeMpiAtExit or bindCalculators, so setting the flag changes
// nothing at exit and the test covers the flag itself: clear at start,
// set after the call, and idempotent. It lives apart from
// CalculatorGroupTest, which mpirun runs on four ranks with the exit
// handler registered; a set flag there would abort that world.
TEST_CASE("Abort flag is clear until a backend requests it",
          "[CalculatorGroup]") {
  REQUIRE_FALSE(rgpot::mpiAbortRequested());
  rgpot::abortMpiAtExit();
  REQUIRE(rgpot::mpiAbortRequested());
  rgpot::abortMpiAtExit();
  REQUIRE(rgpot::mpiAbortRequested());
}
