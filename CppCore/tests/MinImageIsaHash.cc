// MIT License
// Copyright 2023--present rgpot developers

#include "minimage_hash_samples.hpp"

#include <catch2/catch_all.hpp>
#include <limits>

// SHA-256 of the raw little-endian f64 bytes from folded_displacement_hash.
// scalar, SSE2, AVX2 and AVX-512 on this host, and NEON in CI, match it.
// Product-sums are rgpot::simd::mul_add and the file is built with
// contraction off.
static constexpr const char *kFoldHash =
    "d6f5cbaa04ac49e6f02ee64c10eb83d44d951756556924dcc17966453dcd8b11";

TEST_CASE("Folded displacements hash the same on every instruction set",
          "[PairList][isa]") {
  const std::string got = rgpot::nlist::hashcheck::folded_displacement_hash();
  INFO(got);
  REQUIRE(got == kFoldHash);

  const double cell[9] = {4.0, 0.2, 0.1, 0.3, 5.0, -0.2, 0.1, -0.3, 6.0};
  const std::array<bool, 3> periodic{true, true, true};
  const rgpot::nlist::MinimumImage image(cell, periodic);
  double x = std::numeric_limits<double>::quiet_NaN();
  double y = 0.0;
  double z = 0.0;
  REQUIRE_THROWS_AS(image.fold(x, y, z), std::invalid_argument);
}
