#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Loaded-module identity lookup, kept in its own translation unit so tests can
// compile it next to a shared librgpot, which does not export it, and reach
// its error paths without a library the loader refuses to describe. Kept out
// of the installed headers and out of Doxygen.

#include <string>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_EXT_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_EXT_TEST_HIDDEN
#endif

namespace rgpot {

namespace detail {

struct FileIdentity {
  std::string canonical;
#if !defined(_WIN32)
  dev_t dev = 0;
  ino_t ino = 0;
#endif
};

// `path` and a separator to start an error message with, or nothing when no
// path was requested.
[[nodiscard]] inline std::string path_prefix(const std::string &path) {
  return path.empty() ? std::string{} : path + ": ";
}

// Identity of the module mapped at `anchor`. Errors start with
// path_prefix(path).
[[nodiscard]] RGPOT_EXT_TEST_HIDDEN FileIdentity
identify_loaded(const void *anchor, const std::string &path);

} // namespace detail

} // namespace rgpot

#undef RGPOT_EXT_TEST_HIDDEN
