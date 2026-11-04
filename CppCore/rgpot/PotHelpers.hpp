#pragma once
// MIT License
// Copyright 2023--present rgpot developers
#include "ForceStructs.hpp"
#include <atomic>
#include <cstddef>

/**
 * @brief Utility templates and functions for potential management.
 *
 * Defines a static registry of atomic instance and force-call counters.
 * It also provides utility functions for structure initialization and
 * validation.
 */

namespace rgpot {

/**
 * @class registry
 * @brief Static registry for instance tracking and statistics.
 *
 * The registry counts active instances of a derived potential class
 * and how many times those instances have been evaluated.
 */
template <typename T> class registry {
public:
  /// Active-instance and force-call counters. Both are atomic so a batch
  /// worker can construct or evaluate a potential while another worker does
  /// the same (F2). There is no process-global list of instances: each DSO
  /// that includes this header would otherwise keep its own `head`, and the
  /// destructor in librgpot would write through a node freed by the Python
  /// extension.
  static std::atomic<size_t> count; //!< Total number of active instances.
  static std::atomic<size_t>
      forceCalls; //!< Global counter for force evaluations.
  T *prev = nullptr; //!< Unused. Kept so the potential object layout stays.
  T *next = nullptr; //!< Unused. Kept so the potential object layout stays.

protected:
  /**
   * @brief Default constructor.
   */
  registry() { count.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Copy constructor.
   * @param other The instance to copy.
   */
  registry(const registry &) { count.fetch_add(1, std::memory_order_relaxed); }

  /**
   * @brief Destructor.
   */
  ~registry() { count.fetch_sub(1, std::memory_order_relaxed); }

public:
  /**
   * @brief Increments the force call counter.
   * @return Void.
   */
  static void incrementForceCalls() {
    forceCalls.fetch_add(1, std::memory_order_relaxed);
  }
};

template <typename T> std::atomic<size_t> registry<T>::count{0};
template <typename T> std::atomic<size_t> registry<T>::forceCalls{0};

/**
 * @brief Zeroes the members of a ForceOut structure.
 * @param nAtoms The number of atoms.
 * @param efvd The results structure to reset.
 * @return Void.
 */
void zeroForceOut(const size_t &nAtoms, ForceOut *efvd);

/**
 * @brief Validates the input parameters for a potential calculation.
 * @param params The configuration structure to check.
 * @return Void.
 */
void checkParams(const ForceInput &params);

} // namespace rgpot
