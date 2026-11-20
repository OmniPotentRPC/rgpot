#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Field-by-field FNV-1a hashing for potential parameter sets.
 *
 * Potentials fingerprint their configuration into
 * ``PotentialBase::paramsKey()`` so cached results never cross parameter
 * sets. Hash field by field — never memcpy a struct (padding bytes).
 */

#include <bit>
#include <cstdint>
#include <string_view>

namespace rgpot {

struct Fnv1a {
  static constexpr uint64_t kOffset = 0xcbf29ce484222325ULL;
  static constexpr uint64_t kPrime = 0x100000001b3ULL;

  uint64_t h = kOffset;

  constexpr void bytes(const void *data, std::size_t n) noexcept {
    const auto *p = static_cast<const unsigned char *>(data);
    for (std::size_t i = 0; i < n; ++i) {
      h ^= static_cast<uint64_t>(p[i]);
      h *= kPrime;
    }
  }

  void f64(double v) noexcept {
    const auto b = std::bit_cast<uint64_t>(v);
    bytes(&b, sizeof(b));
  }

  void i64(int64_t v) noexcept { bytes(&v, sizeof(v)); }

  void u64(uint64_t v) noexcept { bytes(&v, sizeof(v)); }

  void b8(bool v) noexcept {
    const unsigned char b = v ? 1 : 0;
    bytes(&b, 1);
  }

  void str(std::string_view s) noexcept {
    u64(s.size());
    bytes(s.data(), s.size());
  }
};

} // namespace rgpot
