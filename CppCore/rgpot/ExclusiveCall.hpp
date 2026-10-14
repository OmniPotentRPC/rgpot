#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Detects two threads inside one Reentrancy::PerInstance potential.
 *
 * A potential that keeps per-call state (a torch session, a Verlet pair
 * cache) and advertises PerInstance relies on its callers to keep one
 * instance per thread or per image. This guard costs one atomic exchange
 * and turns a violated contract into an exception instead of a data race.
 */

#include <atomic>
#include <stdexcept>
#include <string>

namespace rgpot {

class ExclusiveCall {
public:
  ExclusiveCall(std::atomic<bool> &busy, const char *who) : busy_(busy) {
    if (busy_.exchange(true, std::memory_order_acquire))
      throw std::logic_error(
          std::string(who) +
          ": one instance was entered by two threads at once; clone one "
          "instance per thread or image (PotCaps::perImageInstances)");
  }
  ~ExclusiveCall() { busy_.store(false, std::memory_order_release); }
  ExclusiveCall(const ExclusiveCall &) = delete;
  ExclusiveCall &operator=(const ExclusiveCall &) = delete;

private:
  std::atomic<bool> &busy_;
};

} // namespace rgpot
