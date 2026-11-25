// MIT License
// Copyright 2023--present rgpot developers
//
// Size and alignment of installed classes whose objects a consumer may
// allocate while the library constructs them. The C++ soversion does not
// change for these, so their layout must not either. Each class is checked
// against a mirror that lists its members as they stand, which holds on
// every platform, and against fixed numbers on x86_64 Linux with
// libstdc++. Thread counts and per-instance buffers added later live
// outside these objects.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/LongRange/Pairs.hpp"

#if defined(RGPOT_HAS_LONG_RANGE)
#include "rgpot/LongRange/Electro.hpp"
#endif

#if defined(__linux__) && defined(__GLIBC__) && defined(__x86_64__) &&        \
    defined(__GLIBCXX__) && defined(__LP64__) && !defined(_GLIBCXX_DEBUG)
#define RGPOT_LAYOUT_PINNED 1
#endif

namespace {

struct DualListMirror {
  double short_cut, long_cut, skin;
  std::unique_ptr<int> short_skin, long_skin;
  std::vector<rgpot::lr::Pair> short_pairs, long_pairs;
  bool short_rebuilt, long_rebuilt, have_vacuum;
  double vacuum[9];
};

template <typename T, typename Mirror> constexpr bool sameShape() {
  return sizeof(T) == sizeof(Mirror) && alignof(T) == alignof(Mirror);
}

static_assert(sameShape<rgpot::lr::DualList, DualListMirror>());

#if defined(RGPOT_LAYOUT_PINNED)
static_assert(sizeof(rgpot::lr::DualList) == 168 &&
              alignof(rgpot::lr::DualList) == 8);
#endif

#if defined(RGPOT_HAS_LONG_RANGE)
struct CoulombPotMirror final : rgpot::Potential<CoulombPotMirror> {
  CoulombPotMirror() : Potential(rgpot::PotType::UNKNOWN) {}
  void forceImpl(const rgpot::ForceInput &, rgpot::ForceOut *) const override {}
  rgpot::lr::CoulombConfig cfg;
  std::shared_ptr<rgpot::lr::ChargeSource> charges;
  std::shared_ptr<rgpot::lr::ChargeResponse> response;
  std::shared_ptr<rgpot::lr::DualList> lists;
  std::uint64_t params = 0;
};

static_assert(sameShape<rgpot::lr::CoulombPot, CoulombPotMirror>());

#if defined(RGPOT_LAYOUT_PINNED)
#if defined(RGPOT_HAS_CACHE)
static_assert(sizeof(rgpot::lr::CoulombPot) == 168);
#else
static_assert(sizeof(rgpot::lr::CoulombPot) == 160);
#endif
static_assert(alignof(rgpot::lr::CoulombPot) == 8);
#endif
#endif

} // namespace

TEST_CASE("installed classes keep their size and alignment", "[abi]") {
  CHECK(sameShape<rgpot::lr::DualList, DualListMirror>());
#if defined(RGPOT_HAS_LONG_RANGE)
  CHECK(sameShape<rgpot::lr::CoulombPot, CoulombPotMirror>());
#endif
}
