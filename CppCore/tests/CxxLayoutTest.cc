// MIT License
// Copyright 2023--present rgpot developers
//
// The C++ layout stamp: what the header computes equals what the library
// exports, the constants describe the types they claim to, and a library
// built from a header with a changed structure is refused.

#include <cstdlib>
#include <string>

#include <catch2/catch_all.hpp>

#include "rgpot/abi/CxxLayout.hpp"
#include "rgpot/plugin/DynLib.hpp"

namespace abi = rgpot::abi;

TEST_CASE("the library stamp equals the header stamp", "[abi][layout]") {
  const auto header = abi::header_stamp();
  const auto library = abi::library_stamp();
  REQUIRE(library.revision == header.revision);
  REQUIRE(library.features == header.features);
  REQUIRE(library.hash == header.hash);
  REQUIRE(abi::layout_compatible());
}

TEST_CASE("the header stamp is a compile-time constant", "[abi][layout]") {
  constexpr auto stamp = abi::header_stamp();
  static_assert(stamp.revision == abi::kCxxLayoutRevision);
  static_assert(stamp.features == abi::kFeatures);
  static_assert(stamp.hash != 0);
  REQUIRE(stamp.hash == abi::header_stamp().hash);
}

TEST_CASE("the feature bit follows RGPOT_HAS_CACHE", "[abi][layout]") {
#ifdef RGPOT_HAS_CACHE
  REQUIRE(abi::header_stamp().features == 1u);
#else
  REQUIRE(abi::header_stamp().features == 0u);
#endif
}

TEST_CASE("the exported check refuses a different stamp", "[abi][layout]") {
  auto stamp = abi::header_stamp();
  REQUIRE(rgpot_cxx_layout_compatible(&stamp) == 1);
  auto other = stamp;
  other.hash ^= 1u;
  REQUIRE(rgpot_cxx_layout_compatible(&other) == 0);
  other = stamp;
  other.revision += 1;
  REQUIRE(rgpot_cxx_layout_compatible(&other) == 0);
  other = stamp;
  other.features ^= 1u;
  REQUIRE(rgpot_cxx_layout_compatible(&other) == 0);
  REQUIRE(rgpot_cxx_layout_compatible(nullptr) == 0);
}

TEST_CASE("a library built with a changed structure is refused",
          "[abi][layout]") {
  const char *path = std::getenv("RGPOT_LAYOUT_FIXTURE");
  REQUIRE(path != nullptr);
  auto handle = rgpot::plugin::dynlib::open(path);
  REQUIRE(handle != nullptr);
  using StampFn = rgpot_cxx_layout_stamp_t (*)(void);
  auto stamp_fn = reinterpret_cast<StampFn>(
      rgpot::plugin::dynlib::sym(handle, "rgpot_cxx_layout_stamp"));
  REQUIRE(stamp_fn != nullptr);
  const auto fixture = stamp_fn();
  const auto mine = abi::header_stamp();
  REQUIRE(fixture.revision == mine.revision);
  REQUIRE(fixture.features == mine.features);
  REQUIRE(fixture.hash != mine.hash);

  using CheckFn = int32_t (*)(const rgpot_cxx_layout_stamp_t *);
  auto check_fn = reinterpret_cast<CheckFn>(
      rgpot::plugin::dynlib::sym(handle, "rgpot_cxx_layout_compatible"));
  REQUIRE(check_fn != nullptr);
  REQUIRE(check_fn(&mine) == 0);
  rgpot::plugin::dynlib::close(handle);
}
