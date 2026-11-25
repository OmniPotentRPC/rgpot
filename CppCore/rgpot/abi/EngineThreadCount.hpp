#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file EngineThreadCount.hpp
 * @brief The thread count a dlopen frontend records for its engine.
 *
 * A frontend that loads @c libxtb_engine.so or @c libmetatomic_engine.so
 * does not run the kernel: the engine does, with the threads of its own
 * OpenMP, BLAS or torch runtime, which do not read the rgpot count. The
 * engine C ABIs (@c RGPOT_XTB_ABI_VERSION and @c RGPOT_MTA_ABI_VERSION, both
 * 1) export no call that takes a thread count, so a count set in the host
 * cannot reach those threads. The frontend therefore accepts
 * @c setThreadCount as a defined no-op: it checks the argument as
 * @c PotentialBase::setThreadCount does, records the value so
 * @c threadCount reports it, and changes nothing about how the engine
 * evaluates. A host that needs fewer engine threads sets them in the
 * engine's own runtime, for example @c OMP_NUM_THREADS in the environment
 * the engine starts in.
 */

#include <atomic>
#include <stdexcept>

namespace rgpot::abi {

class EngineThreadCount {
public:
  EngineThreadCount() = default;
  EngineThreadCount(const EngineThreadCount &) = delete;
  EngineThreadCount &operator=(const EngineThreadCount &) = delete;

  /// Throws std::invalid_argument when @p threads is below 1.
  void set(int threads) {
    if (threads < 1) {
      throw std::invalid_argument("rgpot thread count must be at least 1");
    }
    threads_.store(threads, std::memory_order_relaxed);
  }

  void clear() noexcept { threads_.store(0, std::memory_order_relaxed); }

  /// Zero when no count has been recorded.
  [[nodiscard]] int get() const noexcept {
    return threads_.load(std::memory_order_relaxed);
  }

private:
  std::atomic<int> threads_{0};
};

} // namespace rgpot::abi
