// MIT License
// Copyright 2023--present rgpot developers
//
// One vesin neighbour list rebuilt across geometries with and without
// pairs: a first call with none, pairs then none, none then pairs. The
// systems are empty, a single atom, and two atoms past the cutoff. Every
// algorithm, both list shells, with and without a Verlet skin, on open and
// periodic cells. Each stored entry is read back, so a short allocation
// shows up under AddressSanitizer.

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "vesin.h"

namespace {

VesinOptions options_for(VesinAlgorithm algorithm, double skin, bool full = false) {
  VesinOptions options{};
  options.cutoff = 2.0;
  options.full = full;
  options.sorted = false;
  options.algorithm = algorithm;
  options.skin = skin;
  options.n_threads = 1;
  options.return_shifts = true;
  options.return_distances = true;
  options.return_vectors = true;
  return options;
}

void build(const double (&points)[2][3], VesinOptions options, VesinNeighborList &neighbors) {
  const double box[3][3] = {
      {30.0, 0.0, 0.0},
      {0.0, 30.0, 0.0},
      {0.0, 0.0, 30.0},
  };
  const bool periodic[3] = {false, false, false};
  const char *error = nullptr;
  const VesinDevice device{VesinCPU, 0};
  const int status =
      vesin_neighbors(points, 2, box, periodic, device, options, &neighbors, &error);
  if (status != EXIT_SUCCESS) {
    if (error != nullptr) {
      FAIL(error);
    }
    FAIL("vesin_neighbors failed");
  }
}

struct Geometry {
  const char *name;
  std::vector<double> xyz;
  std::size_t half_pairs;
};

// Two atoms one angstrom apart is the only geometry with a pair. In the
// 30 angstrom box the periodic images of every geometry are past the
// cutoff, so open and periodic cells store the same pairs.
const std::array<Geometry, 4> &geometries() {
  static const std::array<Geometry, 4> all{{
      {"empty", {}, 0},
      {"one atom", {5.0, 5.0, 5.0}, 0},
      {"apart", {5.0, 5.0, 5.0, 15.0, 5.0, 5.0}, 0},
      {"close", {5.0, 5.0, 5.0, 6.0, 5.0, 5.0}, 1},
  }};
  return all;
}

void build_checked(const Geometry &geometry, bool periodic_cell, VesinOptions options,
                   VesinNeighborList &neighbors) {
  const double box[3][3] = {
      {30.0, 0.0, 0.0},
      {0.0, 30.0, 0.0},
      {0.0, 0.0, 30.0},
  };
  const bool periodic[3] = {periodic_cell, periodic_cell, periodic_cell};
  // An empty std::vector may hand out a null data(), and vesin rejects a
  // null points pointer even when n_points is zero.
  static const double origin[1][3] = {{0.0, 0.0, 0.0}};
  const std::size_t n = geometry.xyz.size() / 3;
  const auto *points =
      n == 0 ? origin : reinterpret_cast<const double(*)[3]>(geometry.xyz.data());
  const char *error = nullptr;
  const int status = vesin_neighbors(points, n, box, periodic, VesinDevice{VesinCPU, 0},
                                     options, &neighbors, &error);
  INFO("geometry " << geometry.name);
  REQUIRE(status == EXIT_SUCCESS);
  REQUIRE(error == nullptr);
  const std::size_t expected = geometry.half_pairs * (options.full ? 2 : 1);
  REQUIRE(neighbors.length == expected);
  // A list of length zero still hands out every requested array. Callers
  // that wrap them (Fortran c_f_pointer, a Rust slice) need non-NULL.
  REQUIRE(neighbors.pairs != nullptr);
  REQUIRE(neighbors.shifts != nullptr);
  REQUIRE(neighbors.distances != nullptr);
  REQUIRE(neighbors.vectors != nullptr);
  for (std::size_t k = 0; k < neighbors.length; ++k) {
    const auto i = neighbors.pairs[k][0];
    const auto j = neighbors.pairs[k][1];
    REQUIRE(i < n);
    REQUIRE(j < n);
    REQUIRE(i != j);
    for (int d = 0; d < 3; ++d) {
      REQUIRE(neighbors.shifts[k][d] == 0);
    }
    const double dx = geometry.xyz[3 * j] - geometry.xyz[3 * i];
    REQUIRE(neighbors.vectors[k][0] == dx);
    REQUIRE(neighbors.vectors[k][1] == 0.0);
    REQUIRE(neighbors.vectors[k][2] == 0.0);
    REQUIRE(neighbors.distances[k] == std::abs(dx));
  }
}

} // namespace

TEST_CASE("resetting a vesin neighbour list with zero pairs", "[vesin][nlist]") {
  // Ten angstroms apart, cutoff of two: no pair in either direction.
  const double apart[2][3] = {
      {0.0, 0.0, 0.0},
      {10.0, 0.0, 0.0},
  };
  const double close[2][3] = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
  };

  SECTION("cell list recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinCellList, 0.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("brute force recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinBruteForce, 0.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("skin filter recompute") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinCellList, 1.0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }

  SECTION("a stored pair then none") {
    VesinNeighborList neighbors;
    const auto options = options_for(VesinBruteForce, 0.0);
    build(close, options, neighbors);
    REQUIRE(neighbors.length == 1);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    build(apart, options, neighbors);
    REQUIRE(neighbors.length == 0);
    vesin_free(&neighbors);
  }
}

TEST_CASE("a vesin list rebuilt between zero pairs and pairs", "[vesin][nlist]") {
  const auto algorithm = GENERATE(VesinAutoAlgorithm, VesinBruteForce, VesinCellList);
  const bool full = GENERATE(false, true);
  const double skin = GENERATE(0.0, 1.0);
  const bool periodic = GENERATE(false, true);
  const auto options = options_for(algorithm, skin, full);
  CAPTURE(algorithm, full, skin, periodic);

  const auto &all = geometries();
  const auto &close = all[3];

  SECTION("first call with zero pairs") {
    for (std::size_t g = 0; g < 3; ++g) {
      VesinNeighborList neighbors;
      build_checked(all[g], periodic, options, neighbors);
      build_checked(all[g], periodic, options, neighbors);
      vesin_free(&neighbors);
    }
  }

  SECTION("pairs, then reset to zero pairs") {
    for (std::size_t g = 0; g < 3; ++g) {
      VesinNeighborList neighbors;
      build_checked(close, periodic, options, neighbors);
      build_checked(all[g], periodic, options, neighbors);
      build_checked(all[g], periodic, options, neighbors);
      vesin_free(&neighbors);
    }
  }

  SECTION("zero pairs, then pairs again") {
    for (std::size_t g = 0; g < 3; ++g) {
      VesinNeighborList neighbors;
      build_checked(all[g], periodic, options, neighbors);
      build_checked(close, periodic, options, neighbors);
      build_checked(all[g], periodic, options, neighbors);
      build_checked(close, periodic, options, neighbors);
      vesin_free(&neighbors);
    }
  }
}
