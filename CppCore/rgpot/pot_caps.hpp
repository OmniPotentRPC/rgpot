#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Capability descriptors for potential implementations.
 *
 * Consumers that evaluate potentials from multiple threads (NEB images,
 * dimer endpoints, batched searches) read these instead of maintaining
 * per-potential blocklists.
 */

#include <cstdint>

namespace rgpot {

/**
 * @brief How a potential tolerates concurrent evaluation.
 */
enum class Reentrancy : uint8_t {
  /// One instance may be called from many threads concurrently.
  SharedInstance,
  /// Concurrent evaluation needs one instance per thread (per-instance
  /// native handles or scratch state).
  PerInstance,
  /// Global or COMMON-block state: all evaluations in the process must be
  /// serialized regardless of instance count.
  ProcessSerial,
};

/**
 * @brief Capability set a potential advertises to its callers.
 */
struct PotCaps {
  Reentrancy reentrancy = Reentrancy::SharedInstance;
  /// Callers running multi-image methods (NEB) should clone one instance
  /// per image even when reentrancy alone would allow sharing.
  bool perImageInstances = false;
  /// Native multi-system batch evaluation exists (forceBatch-style hook).
  bool batched = false;
  /// Kernel supports periodic boundary conditions.
  bool periodic = true;
  /// The potential publishes Cauchy stress for a positive cell volume.
  /// Develop's ForceOut has no stress slot, so the tensor is read from the
  /// potential (IPIPot::copyStress) and from the RPC result.
  bool stress = false;
};

} // namespace rgpot
