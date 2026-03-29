#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Parallel batch evaluation of multiple configurations.
 *
 * BatchEvaluator takes a factory function that creates independent potential
 * instances and evaluates N configurations in parallel using a thread pool.
 * Each thread owns its own potential instance (no sharing, no locks).
 *
 * Usage:
 *   auto factory = []() { return std::make_unique<rgpot::LJPot>(); };
 *   rgpot::BatchEvaluator eval(factory, 4); // 4 threads
 *
 *   std::vector<rgpot::BatchInput> inputs = { ... };
 *   auto results = eval.evaluate(inputs);
 *   // results[i].energy, results[i].forces
 */

#include <array>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

struct BatchInput {
  types::AtomMatrix positions;
  std::vector<int> atomic_numbers;
  std::array<std::array<double, 3>, 3> box;
};

struct BatchResult {
  double energy;
  types::AtomMatrix forces;
};

using PotentialFactory = std::function<std::unique_ptr<PotentialBase>()>;

class BatchEvaluator {
public:
  /// Create a batch evaluator with the given factory and thread count.
  /// @param factory  Called once per thread to create an independent instance.
  /// @param n_threads  Number of worker threads (0 = hardware_concurrency).
  BatchEvaluator(PotentialFactory factory, unsigned n_threads = 0);

  /// Evaluate all configurations in parallel and return results.
  /// The order of results matches the order of inputs.
  std::vector<BatchResult> evaluate(const std::vector<BatchInput> &inputs);

  /// Number of worker threads.
  [[nodiscard]] unsigned n_threads() const { return m_n_threads; }

private:
  PotentialFactory m_factory;
  unsigned m_n_threads;
  std::vector<std::unique_ptr<PotentialBase>> m_potentials;
};

} // namespace rgpot
