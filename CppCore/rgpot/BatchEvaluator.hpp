#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Parallel evaluation of independent configurations.
 *
 * A BatchEvaluator owns one potential instance per worker, created by a
 * caller-supplied factory, and splits the configurations of one call into
 * contiguous ranges, one range per worker. Instances share nothing, so no
 * potential has to be re-entrant across threads; a potential that reports
 * @c PotCaps::reentrancy as process-serial is safe here because each
 * instance is touched by one thread at a time.
 *
 * @code
 * rgpot::BatchEvaluator eval(
 *     [] { return std::make_unique<rgpot::LJPot>(); }, 4);
 * auto results = eval.evaluate(inputs);  // results[i] answers inputs[i]
 * @endcode
 *
 * A worker that throws stops its range; evaluate() rethrows the exception
 * of the lowest-numbered failing worker after every worker has joined.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/// One configuration of a batch.
struct BatchInput {
  types::AtomMatrix positions;
  std::vector<int> atomic_numbers;
  std::array<std::array<double, 3>, 3> box;
};

/// Energy, forces and variance of one configuration.
struct BatchResult {
  double energy = 0.0;
  types::AtomMatrix forces;
  double variance = 0.0;
};

/// Creates one independent potential instance.
using PotentialFactory = std::function<std::unique_ptr<PotentialBase>()>;

class BatchEvaluator {
public:
  /**
   * @param factory   Called once per worker; must return a distinct instance
   *                  each call.
   * @param n_threads Number of workers; 0 selects the hardware concurrency
   *                  (at least 1).
   */
  explicit BatchEvaluator(PotentialFactory factory, unsigned n_threads = 0)
      : m_n_threads(n_threads != 0 ? n_threads
                                   : std::max(1u, std::thread::hardware_concurrency())) {
    if (!factory) {
      throw std::invalid_argument("BatchEvaluator: empty potential factory");
    }
    m_potentials.reserve(m_n_threads);
    for (unsigned i = 0; i < m_n_threads; ++i) {
      auto pot = factory();
      if (!pot) {
        throw std::invalid_argument(
            "BatchEvaluator: the factory returned a null potential");
      }
      m_potentials.push_back(std::move(pot));
    }
  }

  /// Evaluate every configuration; results[i] answers inputs[i].
  std::vector<BatchResult> evaluate(const std::vector<BatchInput> &inputs) {
    std::vector<BatchResult> results(inputs.size());
    if (inputs.empty()) {
      return results;
    }
    const std::size_t workers =
        std::min<std::size_t>(m_n_threads, inputs.size());
    std::vector<std::exception_ptr> errors(workers);
    auto run = [&](std::size_t w, std::size_t first, std::size_t last) {
      try {
        for (std::size_t i = first; i < last; ++i) {
          auto [energy, forces, variance] = (*m_potentials[w])(
              inputs[i].positions, inputs[i].atomic_numbers, inputs[i].box);
          results[i] = BatchResult{energy, std::move(forces), variance};
        }
      } catch (...) {
        errors[w] = std::current_exception();
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(workers);
    const std::size_t base = inputs.size() / workers;
    const std::size_t extra = inputs.size() % workers;
    std::size_t first = 0;
    for (std::size_t w = 0; w < workers; ++w) {
      const std::size_t last = first + base + (w < extra ? 1 : 0);
      threads.emplace_back(run, w, first, last);
      first = last;
    }
    for (auto &t : threads) {
      t.join();
    }
    for (auto &e : errors) {
      if (e) {
        std::rethrow_exception(e);
      }
    }
    return results;
  }

  /// Number of workers.
  [[nodiscard]] unsigned n_threads() const noexcept { return m_n_threads; }

private:
  unsigned m_n_threads;
  std::vector<std::unique_ptr<PotentialBase>> m_potentials;
};

} // namespace rgpot
