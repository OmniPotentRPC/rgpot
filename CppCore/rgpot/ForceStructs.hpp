#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief POD structures for force and energy calculation interfaces.
 *
 * Defines the core data exchange structures used between the
 * high-level potential wrappers and the low-level physics engines.
 */

#include <cstddef>

namespace rgpot {

/**
 * @brief Data structure containing input configuration for force calls.
 * @ingroup rgpot
 */
typedef struct {
  const size_t nAtoms; //!< Total number of atoms in the system.
  const double *pos;   //!< Pointer to the flat array of atomic positions.
  const int *atmnrs;   //!< Pointer to the array of atomic numbers.
  const double *box;   //!< Pointer to the 3x3 simulation cell matrix.
} ForceInput;

/**
 * @brief Data structure to store results from force calculations.
 * @ingroup rgpot
 */
typedef struct {
  double *F;       //!< Pointer to the array where forces will be stored.
  double energy;   //!< Calculated potential energy of the configuration.
  double variance; //!< Variance or uncertainty of the calculation.
  // Variance here is 0 when not needed and that's OK
} ForceOut;

} // namespace rgpot
