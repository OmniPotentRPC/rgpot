// MIT License
// Copyright 2023--present rgpot developers

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include "rgpot/BatchEvaluator.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

using rgpot::types::AtomMatrix;
using Catch::Matchers::WithinAbs;

static rgpot::BatchInput make_two_atom(double separation) {
  AtomMatrix pos(2, 3);
  pos(0, 0) = 0.0;
  pos(0, 1) = 0.0;
  pos(0, 2) = 0.0;
  pos(1, 0) = separation;
  pos(1, 1) = 0.0;
  pos(1, 2) = 0.0;
  return {pos, {1, 1}, {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}}};
}

TEST_CASE("BatchEvaluator serial matches direct", "[batch]") {
  auto factory = []() -> std::unique_ptr<rgpot::PotentialBase> {
    return std::make_unique<rgpot::LJPot>();
  };

  rgpot::BatchEvaluator eval(factory, 1);

  std::vector<rgpot::BatchInput> inputs;
  for (double r = 1.0; r <= 3.0; r += 0.5) {
    inputs.push_back(make_two_atom(r));
  }

  auto results = eval.evaluate(inputs);
  REQUIRE(results.size() == inputs.size());

  // Compare with direct evaluation
  rgpot::LJPot direct;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto [de, df] =
        direct(inputs[i].positions, inputs[i].atomic_numbers, inputs[i].box);
    REQUIRE_THAT(results[i].energy, WithinAbs(de, 1e-12));
  }
}

TEST_CASE("BatchEvaluator parallel matches serial", "[batch]") {
  auto factory = []() -> std::unique_ptr<rgpot::PotentialBase> {
    return std::make_unique<rgpot::LJPot>();
  };

  // Generate 20 configs at different separations
  std::vector<rgpot::BatchInput> inputs;
  for (double r = 1.0; r <= 5.0; r += 0.2) {
    inputs.push_back(make_two_atom(r));
  }

  // Serial
  rgpot::BatchEvaluator serial(factory, 1);
  auto serial_results = serial.evaluate(inputs);

  // Parallel (4 threads)
  rgpot::BatchEvaluator parallel(factory, 4);
  auto parallel_results = parallel.evaluate(inputs);

  REQUIRE(serial_results.size() == parallel_results.size());
  for (size_t i = 0; i < serial_results.size(); ++i) {
    REQUIRE_THAT(parallel_results[i].energy,
                 WithinAbs(serial_results[i].energy, 1e-12));
    for (size_t a = 0; a < 2; ++a) {
      for (size_t c = 0; c < 3; ++c) {
        REQUIRE_THAT(parallel_results[i].forces(a, c),
                     WithinAbs(serial_results[i].forces(a, c), 1e-12));
      }
    }
  }
}

TEST_CASE("BatchEvaluator empty input", "[batch]") {
  auto factory = []() -> std::unique_ptr<rgpot::PotentialBase> {
    return std::make_unique<rgpot::LJPot>();
  };
  rgpot::BatchEvaluator eval(factory, 2);
  auto results = eval.evaluate({});
  REQUIRE(results.empty());
}
