// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/BatchEvaluator.hpp"

#include <algorithm>
#include <thread>

namespace rgpot {

BatchEvaluator::BatchEvaluator(PotentialFactory factory, unsigned n_threads)
    : m_factory(std::move(factory)),
      m_n_threads(n_threads > 0 ? n_threads
                                : std::thread::hardware_concurrency()) {
  if (m_n_threads == 0)
    m_n_threads = 1;

  // Pre-create one potential per thread
  m_potentials.reserve(m_n_threads);
  for (unsigned i = 0; i < m_n_threads; ++i) {
    m_potentials.push_back(m_factory());
  }
}

std::vector<BatchResult>
BatchEvaluator::evaluate(const std::vector<BatchInput> &inputs) {
  const size_t n = inputs.size();
  std::vector<BatchResult> results(n);

  if (n == 0)
    return results;

  // For single input or single thread, evaluate serially
  if (n == 1 || m_n_threads == 1) {
    for (size_t i = 0; i < n; ++i) {
      auto [energy, forces] = (*m_potentials[0])(
          inputs[i].positions, inputs[i].atomic_numbers, inputs[i].box);
      results[i] = {energy, std::move(forces)};
    }
    return results;
  }

  // Parallel evaluation: partition work across threads
  auto worker = [&](unsigned tid, size_t begin, size_t end) {
    auto &pot = *m_potentials[tid];
    for (size_t i = begin; i < end; ++i) {
      auto [energy, forces] =
          pot(inputs[i].positions, inputs[i].atomic_numbers, inputs[i].box);
      results[i] = {energy, std::move(forces)};
    }
  };

  // Simple static partitioning
  std::vector<std::thread> threads;
  threads.reserve(m_n_threads);
  size_t chunk = (n + m_n_threads - 1) / m_n_threads;

  for (unsigned t = 0; t < m_n_threads; ++t) {
    size_t begin = t * chunk;
    size_t end = std::min(begin + chunk, n);
    if (begin >= n)
      break;
    threads.emplace_back(worker, t, begin, end);
  }

  for (auto &t : threads) {
    t.join();
  }

  return results;
}

} // namespace rgpot
