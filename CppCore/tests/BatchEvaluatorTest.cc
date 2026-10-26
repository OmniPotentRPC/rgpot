// MIT License
// Copyright 2023--present rgpot developers

#include <array>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/BatchEvaluator.hpp"
#include "rgpot/LennardJones/LJPot.hpp"

using rgpot::types::AtomMatrix;

namespace {

rgpot::BatchInput two_atoms(double separation) {
  AtomMatrix pos = AtomMatrix::Zero(2, 3);
  pos(1, 0) = separation;
  return {pos, {1, 1}, {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}}};
}

std::vector<rgpot::BatchInput> ladder(std::size_t n) {
  std::vector<rgpot::BatchInput> inputs;
  for (std::size_t i = 0; i < n; ++i) {
    inputs.push_back(two_atoms(1.0 + 0.2 * static_cast<double>(i)));
  }
  return inputs;
}

bool same_matrix(const AtomMatrix &a, const AtomMatrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a.data()[i] != b.data()[i]) {
      return false;
    }
  }
  return true;
}

/// Throws on the third call of the instance, returns zeros before it.
class FlakyPot : public rgpot::PotentialBase {
public:
  FlakyPot() : PotentialBase(rgpot::PotType::LJ) {}
  std::tuple<double, AtomMatrix, double>
  operator()(const AtomMatrix &positions, const std::vector<int> &,
             const std::array<std::array<double, 3>, 3> &) override {
    if (++m_calls == 3) {
      throw std::runtime_error("flaky potential failed");
    }
    return {0.0, AtomMatrix::Zero(positions.rows(), 3), 0.0};
  }

private:
  int m_calls = 0;
};

rgpot::PotentialFactory lj_factory() {
  return [] { return std::make_unique<rgpot::LJPot>(); };
}

} // namespace

TEST_CASE("BatchEvaluator one worker equals direct calls", "[batch]") {
  const auto inputs = ladder(9);
  rgpot::BatchEvaluator eval(lj_factory(), 1);
  const auto results = eval.evaluate(inputs);
  REQUIRE(results.size() == inputs.size());
  rgpot::LJPot direct;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto [energy, forces, variance] =
        direct(inputs[i].positions, inputs[i].atomic_numbers, inputs[i].box);
    REQUIRE(results[i].energy == energy);
    REQUIRE(results[i].variance == variance);
    REQUIRE(same_matrix(results[i].forces, forces));
  }
}

TEST_CASE("BatchEvaluator workers keep the input order", "[batch]") {
  // 21 inputs over 4 workers: ranges of 6, 5, 5, 5.
  const auto inputs = ladder(21);
  rgpot::BatchEvaluator serial(lj_factory(), 1);
  rgpot::BatchEvaluator parallel(lj_factory(), 4);
  REQUIRE(parallel.n_threads() == 4);
  const auto a = serial.evaluate(inputs);
  const auto b = parallel.evaluate(inputs);
  REQUIRE(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    REQUIRE(b[i].energy == a[i].energy);
    REQUIRE(same_matrix(b[i].forces, a[i].forces));
  }
}

TEST_CASE("BatchEvaluator with more workers than inputs", "[batch]") {
  const auto inputs = ladder(2);
  rgpot::BatchEvaluator eval(lj_factory(), 8);
  REQUIRE(eval.evaluate(inputs).size() == 2);
}

TEST_CASE("BatchEvaluator empty input", "[batch]") {
  rgpot::BatchEvaluator eval(lj_factory(), 2);
  REQUIRE(eval.evaluate({}).empty());
}

TEST_CASE("BatchEvaluator zero threads selects at least one worker", "[batch]") {
  rgpot::BatchEvaluator eval(lj_factory(), 0);
  REQUIRE(eval.n_threads() >= 1);
}

TEST_CASE("BatchEvaluator refuses a bad factory", "[batch]") {
  REQUIRE_THROWS_AS(rgpot::BatchEvaluator(rgpot::PotentialFactory{}, 1),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::BatchEvaluator(
                        [] { return std::unique_ptr<rgpot::PotentialBase>(); },
                        1),
                    std::invalid_argument);
}

TEST_CASE("BatchEvaluator rethrows a worker failure after joining", "[batch]") {
  rgpot::BatchEvaluator eval([] { return std::make_unique<FlakyPot>(); }, 2);
  // Each worker evaluates four inputs; the third call of each instance throws.
  REQUIRE_THROWS_WITH(eval.evaluate(ladder(8)),
                      Catch::Matchers::ContainsSubstring("flaky"));
}
