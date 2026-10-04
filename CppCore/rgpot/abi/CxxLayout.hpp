#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file CxxLayout.hpp
 * @brief Layout stamp of the C++ types that cross the librgpot boundary.
 *
 * The header-only classes (@c Potential, the force structures, the pair
 * potentials) are compiled into the consumer, while their out-of-line members
 * live in the library. A library built from another revision of these
 * headers, or with another value of @c RGPOT_HAS_CACHE, runs with the wrong
 * member offsets or the wrong vtable and fails far from the cause.
 *
 * The stamp folds @c sizeof, @c alignof and @c offsetof of every such type
 * into one 64 bit hash. The hash is a @c constexpr of this header, so a
 * consumer holds what it compiled against; the library exports the same
 * computation through @c rgpot_cxx_layout_stamp so the consumer can read what
 * it loaded.
 *
 * @code
 * if (!rgpot::abi::layout_compatible()) {
 *   // refuse to run: the library differs from these headers
 * }
 * @endcode
 *
 * The covered types, the rule for @c kCxxLayoutRevision and the
 * virtual-function counters are documented in
 * @c docs/orgmode/reference/c_abi.org.
 */

#include <cstddef>
#include <cstdint>

#include "rgpot/ForceStructs.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/CalculatorGroup.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/pot_types.hpp"

#ifdef RGPOT_HAS_CACHE
#include "rgpot/PotentialCache.hpp"
#endif

#if defined(_WIN32) && defined(RGPOT_CXX_LAYOUT_BUILDING)
#define RGPOT_CXX_LAYOUT_API __declspec(dllexport)
#else
#define RGPOT_CXX_LAYOUT_API
#endif

/// Plain-C view of the stamp, returned by the exported function.
extern "C" {
typedef struct rgpot_cxx_layout_stamp_t {
  uint32_t revision; ///< kCxxLayoutRevision of the header that built it.
  uint32_t features; ///< Bit 0: RGPOT_HAS_CACHE.
  uint64_t hash;     ///< Fold of sizeof/alignof/offsetof of the covered types.
} rgpot_cxx_layout_stamp_t;

/// Stamp of the loaded library, computed from the headers it was built with.
RGPOT_CXX_LAYOUT_API rgpot_cxx_layout_stamp_t rgpot_cxx_layout_stamp(void);

/// Nonzero when @p caller equals the stamp of the loaded library.
RGPOT_CXX_LAYOUT_API int32_t
rgpot_cxx_layout_compatible(const rgpot_cxx_layout_stamp_t *caller);
}

namespace rgpot::abi {

/// Hand-maintained. Increment when a change the hash cannot see alters the
/// C++ ABI: reordering virtuals without changing their count, changing the
/// meaning of a field, changing a base class of a covered type. Any change to
/// size, alignment or a field offset moves the hash by itself.
inline constexpr std::uint32_t kCxxLayoutRevision = 1;

/// Virtual functions declared by @c PotentialBase, destructor included.
/// The cache hook @c set_cache adds one when @c RGPOT_HAS_CACHE is defined.
/// A test counts the @c virtual declarations in Potential.hpp and fails when
/// this and the header disagree.
inline constexpr std::uint32_t kPotentialBaseVirtuals = 5;
inline constexpr std::uint32_t kPotentialBaseCacheVirtuals = 1;
/// Virtual functions @c Potential<Derived> declares itself (@c forceImpl and
/// @c forceBatchImpl).
inline constexpr std::uint32_t kPotentialVirtuals = 2;

#ifdef RGPOT_HAS_CACHE
inline constexpr std::uint32_t kFeatures = 1;
#else
inline constexpr std::uint32_t kFeatures = 0;
#endif

namespace detail {

/// A complete derived type: gives the CRTP template one instantiation whose
/// size and alignment the stamp covers. Complete because compilers that
/// instantiate the virtual members with the class need the base-to-derived
/// conversion to be valid.
struct LayoutProbe final : Potential<LayoutProbe> {
  LayoutProbe() : Potential(PotType::UNKNOWN) {}
  void forceImpl(const ForceInput &, ForceOut *) const override {}
};

struct Fold {
  std::uint64_t h = 0xcbf29ce484222325ull;
  constexpr void add(std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      h ^= (v >> (8 * i)) & 0xffu;
      h *= 0x100000001b3ull;
    }
  }
  /// sizeof and alignof of a type.
  template <typename T> constexpr void type() {
    add(sizeof(T));
    add(alignof(T));
  }
};

constexpr std::uint64_t compute_hash() {
  Fold f;
  f.add(kCxxLayoutRevision);
  f.add(kFeatures);
  f.add(sizeof(void *));
  f.add(sizeof(std::size_t));

  // Plain structures with a fixed field order.
  f.type<ForceInput>();
  f.add(offsetof(ForceInput, nAtoms));
  f.add(offsetof(ForceInput, pos));
  f.add(offsetof(ForceInput, atmnrs));
  f.add(offsetof(ForceInput, box));
  f.type<ForceOut>();
  f.add(offsetof(ForceOut, F));
  f.add(offsetof(ForceOut, energy));
  f.add(offsetof(ForceOut, variance));
  f.add(offsetof(ForceOut, stress));
  f.add(offsetof(ForceOut, has_stress));
  f.type<ForceBatch>();
  f.add(offsetof(ForceBatch, nSystems));
  f.add(offsetof(ForceBatch, in));
  f.add(offsetof(ForceBatch, out));
  f.type<PotCaps>();
  f.add(offsetof(PotCaps, reentrancy));
  f.add(offsetof(PotCaps, perImageInstances));
  f.add(offsetof(PotCaps, batched));
  f.add(offsetof(PotCaps, periodic));
  f.add(offsetof(PotCaps, groupCollective));
  f.add(offsetof(PotCaps, stress));
  f.type<CalculatorGroup>();
  f.add(offsetof(CalculatorGroup, index));
  f.add(offsetof(CalculatorGroup, ranks));
  f.add(offsetof(CalculatorGroup, rank_in_group));
  f.add(offsetof(CalculatorGroup, world_size));
  f.type<LJConfig>();
  f.add(offsetof(LJConfig, u0));
  f.add(offsetof(LJConfig, cutoff));
  f.add(offsetof(LJConfig, psi));
  f.add(offsetof(LJConfig, switch_width));
  f.type<LJClusterConfig>();
  f.add(offsetof(LJClusterConfig, u0));
  f.add(offsetof(LJClusterConfig, cutoff));
  f.add(offsetof(LJClusterConfig, psi));
  f.add(offsetof(LJClusterConfig, switch_width));
  f.type<MorseConfig>();
  f.add(offsetof(MorseConfig, De));
  f.add(offsetof(MorseConfig, a));
  f.add(offsetof(MorseConfig, re));
  f.add(offsetof(MorseConfig, cutoff));
  f.add(offsetof(MorseConfig, switch_width));
  f.type<ZBLConfig>();
  f.add(offsetof(ZBLConfig, cut_inner));
  f.add(offsetof(ZBLConfig, cut_global));

  // Enumerations: underlying width and the values consumers switch on.
  f.type<Reentrancy>();
  f.add(static_cast<std::uint64_t>(Reentrancy::SharedInstance));
  f.add(static_cast<std::uint64_t>(Reentrancy::PerInstance));
  f.add(static_cast<std::uint64_t>(Reentrancy::ProcessSerial));
  f.type<PotType>();
  f.add(static_cast<std::uint64_t>(PotType::UNKNOWN));
  f.add(static_cast<std::uint64_t>(PotType::CuH2));
  f.add(static_cast<std::uint64_t>(PotType::LJ));
  f.add(static_cast<std::uint64_t>(PotType::Morse));
  f.add(static_cast<std::uint64_t>(PotType::LJCluster));
  f.add(static_cast<std::uint64_t>(PotType::ZBL));

  // Polymorphic classes: size, alignment and the virtual-function counts.
  // offsetof is not defined for them.
  f.type<PotentialBase>();
  f.add(kPotentialBaseVirtuals +
        (kFeatures != 0 ? kPotentialBaseCacheVirtuals : 0u));
  f.type<Potential<LayoutProbe>>();
  f.add(kPotentialVirtuals);
  f.type<LJPot>();
  f.type<LJClusterPot>();
  f.type<MorsePot>();
  f.type<ZBLPot>();
#ifdef RGPOT_HAS_CACHE
  f.type<cache::EvalCounts>();
  f.add(offsetof(cache::EvalCounts, computed));
  f.add(offsetof(cache::EvalCounts, served));
  f.type<cache::PotentialCache>();
#endif
  return f.h;
}

} // namespace detail

/// What the consumer compiled against. Initialise a @c constexpr variable
/// with it; a call evaluated at run time is not tied to this translation
/// unit (see layout_compatible()).
inline constexpr rgpot_cxx_layout_stamp_t header_stamp() noexcept {
  return {kCxxLayoutRevision, kFeatures, detail::compute_hash()};
}

/// What the loaded library was built with.
inline rgpot_cxx_layout_stamp_t library_stamp() noexcept {
  return rgpot_cxx_layout_stamp();
}

/// True when the loaded library was built from the same layouts as these
/// headers: equal revision, equal feature bits, equal hash.
inline bool layout_compatible() noexcept {
  // Evaluated at compile time: a runtime call to an inline function would
  // bind, across shared objects, to whichever definition the loader meets
  // first, and defeat the comparison.
  constexpr rgpot_cxx_layout_stamp_t mine = header_stamp();
  return rgpot_cxx_layout_compatible(&mine) != 0;
}

} // namespace rgpot::abi
