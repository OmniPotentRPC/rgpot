// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of utility functions for potential management.
 *
 * Contains the implementations for global helper functions used to manage and
 * validate the core force and energy data structures.
 */

#include "PotHelpers.hpp"
#include <stdexcept>

namespace rgpot {

/**
 * @details
 * This function performs a manual reset of the @c ForceOut structure.
 * It ensures the energy and variance are set to zero and iterates
 * through the force array to clear components for each atom.
 */
void zeroForceOut(const size_t &nAtoms, ForceOut *efvd) {
  efvd->energy = 0;
  efvd->variance = 0;
  for (size_t idx{0}; idx < nAtoms; idx++) {
    efvd->F[3 * idx] = 0;
    efvd->F[3 * idx + 1] = 0;
    efvd->F[3 * idx + 2] = 0;
  }
}

/**
 * @details
 * Verifies that the input parameters represent a physically valid
 * configuration. Currently, it strictly checks that the system
 * contains at least one atom.
 *
 * @warning Throws a @c std::runtime_error if @a nAtoms is zero or less.
 */
void checkParams(const ForceInput &params) {
  // Simple sanity check
  if (params.nAtoms <= 0) {
    throw std::runtime_error("Can't work with zero atoms in force call");
  }
}

} // namespace rgpot
