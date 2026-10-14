// MIT License
// Copyright 2023--present rgpot developers
//
// ExclusiveCall: a second thread entering a PerInstance potential that is
// already busy gets an exception, and the flag clears on scope exit.

#include <atomic>
#include <future>
#include <stdexcept>

#include <catch2/catch_all.hpp>

#include "rgpot/ExclusiveCall.hpp"

TEST_CASE("ExclusiveCall refuses a concurrent entry", "[caps]") {
  std::atomic<bool> busy{false};
  std::promise<void> entered, release;
  auto holder = std::async(std::launch::async, [&] {
    const rgpot::ExclusiveCall guard(busy, "Holder");
    entered.set_value();
    release.get_future().wait();
  });
  entered.get_future().wait();
  REQUIRE_THROWS_AS(rgpot::ExclusiveCall(busy, "Intruder"), std::logic_error);
  release.set_value();
  holder.get();
  REQUIRE_FALSE(busy.load());
  REQUIRE_NOTHROW(rgpot::ExclusiveCall(busy, "After"));
}

TEST_CASE("ExclusiveCall clears the flag when the call throws", "[caps]") {
  std::atomic<bool> busy{false};
  try {
    const rgpot::ExclusiveCall guard(busy, "Thrower");
    throw std::runtime_error("model failed");
  } catch (const std::runtime_error &) {
  }
  REQUIRE_FALSE(busy.load());
}
