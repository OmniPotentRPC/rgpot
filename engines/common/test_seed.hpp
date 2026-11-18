#pragma once
// MIT License
// Copyright 2026 Rohit Goswami

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace rgpot::test {

/// Parse RGPOT_TEST_SEED as an unsigned 64-bit decimal integer, returning
/// `fallback` when unset. Rejects empty strings, whitespace, signs, non-digits,
/// and uint64_t overflow.
inline std::uint64_t test_seed(std::uint64_t fallback) {
  const char *value = std::getenv("RGPOT_TEST_SEED");
  if (value == nullptr) {
    return fallback;
  }
  if (value[0] == '\0') {
    throw std::runtime_error(
        "RGPOT_TEST_SEED must be an unsigned 64-bit integer, got empty string");
  }
  for (const char *p = value; *p != '\0'; ++p) {
    if (std::isdigit(static_cast<unsigned char>(*p)) == 0) {
      throw std::runtime_error(
          std::string("RGPOT_TEST_SEED must contain only decimal digits "
                      "(no whitespace or sign), got '") +
          value + "'");
    }
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error(
        std::string("RGPOT_TEST_SEED overflows uint64_t: '") + value + "'");
  }
  return static_cast<std::uint64_t>(parsed);
}

} // namespace rgpot::test
