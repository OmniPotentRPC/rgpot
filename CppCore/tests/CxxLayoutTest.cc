// MIT License
// Copyright 2023--present rgpot developers
//
// The C++ layout stamp: what the header computes equals what the library
// exports, the constants describe the types they claim to, and a library
// built from a header with a changed structure is refused.

#include <cstddef>
#include <cstdlib>
#include <string>

#include <catch2/catch_all.hpp>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/abi/CxxLayout.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/plugin/DynLib.hpp"

namespace abi = rgpot::abi;

TEST_CASE("the library stamp equals the header stamp", "[abi][layout]") {
  constexpr auto header = abi::header_stamp();
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

TEST_CASE(
    "the soversion index finds a listed platform and rejects an absent one",
    "[abi][layout]") {
  constexpr int present =
      abi::find_soversion_index("linux-x86_64-libstdc++-lp64");
  constexpr int absent = abi::find_soversion_index("not-a-platform");
  static_assert(present >= 0);
  static_assert(absent < 0);
  REQUIRE(present >= 0);
  REQUIRE(absent < 0);
  REQUIRE(abi::cxx_tag_eq(
      abi::kCxxSoversionTable[static_cast<std::size_t>(present)].tag,
      "linux-x86_64-libstdc++-lp64"));
  constexpr int here = abi::find_soversion_index(abi::kCxxPlatformTag);
  if (here < 0) {
    REQUIRE(abi::recorded_layout_hash() == 0u);
  } else {
    REQUIRE(abi::recorded_layout_hash() ==
            abi::kCxxSoversionTable[static_cast<std::size_t>(here)].hash);
  }
}

TEST_CASE("the recorded soversion hash matches this platform",
          "[abi][layout]") {
  REQUIRE(abi::kCxxSoversion > 0u);
#ifndef RGPOT_HAS_CACHE
  constexpr auto stamp = abi::header_stamp();
  REQUIRE(stamp.features == 0u);
  REQUIRE(stamp.revision == abi::kCxxLayoutRevision);
  constexpr auto recorded = abi::recorded_layout_hash();
  if (recorded == 0u) {
    WARN("no kCxxSoversionTable entry for "
         << abi::kCxxPlatformTag << " hash " << stamp.hash);
  } else {
    REQUIRE(stamp.hash == recorded);
  }
#endif
}

TEST_CASE("the feature bit follows RGPOT_HAS_CACHE", "[abi][layout]") {
#ifdef RGPOT_HAS_CACHE
  REQUIRE(abi::header_stamp().features == 1u);
#else
  REQUIRE(abi::header_stamp().features == 0u);
#endif
}

TEST_CASE("the exported check refuses a different stamp", "[abi][layout]") {
  constexpr rgpot_cxx_layout_stamp_t kHeader = abi::header_stamp();
  rgpot_cxx_layout_stamp_t stamp = kHeader;
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
  constexpr auto mine = abi::header_stamp();
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

TEST_CASE("Neighbours generations share one counter across shared objects",
          "[abi][layout][generation]") {
  if (std::getenv("RGPOT_SHARED_LIBRARY") == nullptr) {
    SKIP("the generation counter boundary requires shared librgpot");
  }
  const char *path = std::getenv("RGPOT_LAYOUT_FIXTURE");
  REQUIRE(path != nullptr);
  auto handle = rgpot::plugin::dynlib::open(path);
  REQUIRE(handle != nullptr);
  using GenerationFn = std::uint64_t (*)(void);
  auto fixture_generation =
      reinterpret_cast<GenerationFn>(rgpot::plugin::dynlib::sym(
          handle, "rgpot_fixture_neighbours_generation"));
  REQUIRE(fixture_generation != nullptr);

  const auto executable_first = rgpot::nlist::Neighbours{}.generation();
  const auto fixture_first = fixture_generation();
  const auto executable_second = rgpot::nlist::Neighbours{}.generation();
  const auto fixture_second = fixture_generation();
  CHECK(executable_first < fixture_first);
  CHECK(fixture_first < executable_second);
  CHECK(executable_second < fixture_second);

  rgpot::plugin::dynlib::close(handle);
}

#if defined(__LP64__) || defined(_WIN64)
// The stress request is the only field soversion 9 adds to ForceInput. It sits
// after the four members older callers fill, so their offsets stay put, and
// ForceOut keeps the layout it had before the request existed.
TEST_CASE("ForceInput only appends the stress request", "[abi][layout]") {
  using rgpot::ForceInput;
  using rgpot::ForceOut;
  STATIC_REQUIRE(offsetof(ForceInput, nAtoms) == 0);
  STATIC_REQUIRE(offsetof(ForceInput, pos) == 8);
  STATIC_REQUIRE(offsetof(ForceInput, atmnrs) == 16);
  STATIC_REQUIRE(offsetof(ForceInput, box) == 24);
  STATIC_REQUIRE(offsetof(ForceInput, compute_stress) == 32);
  STATIC_REQUIRE(sizeof(ForceInput) == 40);
  STATIC_REQUIRE(offsetof(ForceOut, F) == 0);
  STATIC_REQUIRE(offsetof(ForceOut, energy) == 8);
  STATIC_REQUIRE(offsetof(ForceOut, variance) == 16);
  STATIC_REQUIRE(offsetof(ForceOut, stress) == 24);
  STATIC_REQUIRE(offsetof(ForceOut, has_stress) == 96);
  STATIC_REQUIRE(sizeof(ForceOut) == 104);
  REQUIRE_FALSE(ForceInput{0, nullptr, nullptr, nullptr}.compute_stress);
}
#endif
