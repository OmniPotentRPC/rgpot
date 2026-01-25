#pragma once
// MIT License
// Copyright 2023--present rgpot developers
#include "ForceStructs.hpp"
#include <cstddef>

/**
 * @brief Utility templates and functions for potential management.
 *
 * Defines a static registry for tracking potential instances and global force
 * call counters. It also provides utility functions for structure
 * initialization and validation.
 */

namespace rgpot {

/**
 * @class registry
 * @brief Static registry for instance tracking and statistics.
 *
 * The registry uses a linked list to track all active instances
 * of a derived potential class.
 */
template <typename T> class registry {
public:
  static size_t count;      //!< Total number of active instances.
  static size_t forceCalls; //!< Global counter for force evaluations.
  static T *head;           //!< Pointer to the head of the linked list.
  T *prev;                  //!< Pointer to the previous instance in the list.
  T *next;                  //!< Pointer to the next instance in the list.

protected:
  /**
   * @brief Default constructor.
   */
  registry() {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<T *>(this);
    if (next) {
      next->prev = head;
    }
  }

  /**
   * @brief Copy constructor.
   * @param other The instance to copy.
   */
  registry(const registry &) {
    ++count;
    prev = nullptr;
    next = head;
    head = static_cast<T *>(this);
    if (next) {
      next->prev = head;
    }
  }

  /**
   * @brief Destructor.
   */
  ~registry() {
    --count;
    if (prev) {
      prev->next = next;
    }
    if (next) {
      next->prev = prev;
    }
    if (head == this) {
      head = next;
    }
  }

public:
  /**
   * @brief Increments the force call counter.
   * @return Void.
   */
  static void incrementForceCalls() { ++forceCalls; }
};

template <typename T> size_t registry<T>::count = 0;
template <typename T> size_t registry<T>::forceCalls = 0;
template <typename T> T *registry<T>::head = nullptr;

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
