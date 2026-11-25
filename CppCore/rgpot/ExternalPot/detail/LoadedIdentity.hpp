#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Private access to the loaded-module identity lookup in ExternalPot.cc, so
// tests can reach its error paths without a library the loader refuses to
// describe. Kept out of the installed headers and out of Doxygen.

#include <string>

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_EXT_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_EXT_TEST_HIDDEN
#endif

namespace rgpot {

namespace detail {

// Canonical path of the module mapped at `anchor`. Errors name `path` first,
// or start with the reason when `path` is empty.
[[nodiscard]] RGPOT_EXT_TEST_HIDDEN std::string
loaded_canonical(const void *anchor, const std::string &path);

} // namespace detail

} // namespace rgpot

#undef RGPOT_EXT_TEST_HIDDEN
