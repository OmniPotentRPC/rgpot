#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Private snapshot of the thread-local Sutton-Chen stored-list workspace.
// Kept out of the installed SuttonChenPot.hpp header and excluded from
// Doxygen and meson/CMake header installation so public consumers only see
// the SuttonChenPot interface.

#include <cstddef>
#include <cstdint>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_SC_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_SC_TEST_HIDDEN
#endif

namespace rgpot {

namespace detail {

struct SuttonChenWorkspaceSnapshot {
  std::uint64_t generation = 0;
  std::uint64_t index_builds = 0;
  std::size_t atoms = 0;
  std::size_t total_up = 0;
  std::size_t index_capacity_bytes = 0;
  std::size_t scratch_capacity_bytes = 0;
};

[[nodiscard]] RGPOT_SC_TEST_HIDDEN SuttonChenWorkspaceSnapshot
suttonChenWorkspaceSnapshot();

RGPOT_SC_TEST_HIDDEN void releaseSuttonChenWorkspace() noexcept;

} // namespace detail

} // namespace rgpot

#undef RGPOT_SC_TEST_HIDDEN
