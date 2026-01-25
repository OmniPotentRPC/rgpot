// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Lennard-Jones potential methods.
 *
 * This file contains the implementation of the force and energy
 * calculation for the Lennard-Jones potential, including periodic
 * boundary condition handling.
 */

// clang-format off
#include <cmath>
#include <limits>
// clang-format on

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @details Implementation of a shifted 12-6 Lennard-Jones potential.
 *
 * This method calculates pairwise interactions between all atoms
 * within the cutoff radius. It applies the minimum image convention
 * using the provided box dimensions to handle periodic boundaries.
 *
 * @note This implementation is adapted, untouched from the eOn project [1].
 * @warning The box is assumed to be orthogonal.
 *
 * # References
 * [1] EON Development Team. LJ.cpp.
 * https://github.com/TheochemUI/EONgit/blob/stable/client/potentials/LJ/LJ.cpp
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  // This is adapted, untouched from EON's BSD 3 clause implementation
  // Original source:
  // https://github.com/TheochemUI/EONgit/blob/stable/client/potentials/LJ/LJ.cpp
  // Copyright (c) 2010, EON Development Team
  // All rights reserved. BSD 3-Clause License.
  double diffR{0}, diffRX{0}, diffRY{0}, diffRZ{0}, dU{0}, a{0}, b{0};
  *U = 0;
  for (int i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }

  for (int i = 0; i < N - 1; i++) {
    for (int j = i + 1; j < N; j++) {
      diffRX = R[3 * i] - R[3 * j];
      diffRY = R[3 * i + 1] - R[3 * j + 1];
      diffRZ = R[3 * i + 2] - R[3 * j + 2];

      // Minimum image convention
      diffRX = diffRX - box[0] * floor(diffRX / box[0] + 0.5);
      diffRY = diffRY - box[4] * floor(diffRY / box[4] + 0.5);
      diffRZ = diffRZ - box[8] * floor(diffRZ / box[8] + 0.5);

      diffR = sqrt(diffRX * diffRX + diffRY * diffRY + diffRZ * diffRZ);

      if (diffR < cuttOffR) {
        // Standard 12-6 form: 4u0((psi/r)^12 - (psi/r)^6)
        a = pow(psi / diffR, 6);
        b = 4 * u0 * a;

        *U = *U + b * (a - 1) - cuttOffU;

        dU = -6 * b / diffR * (2 * a - 1);

        // Update forces for both atoms
        // F is the negative derivative
        F[3 * i] = F[3 * i] - dU * diffRX / diffR;
        F[3 * i + 1] = F[3 * i + 1] - dU * diffRY / diffR;
        F[3 * i + 2] = F[3 * i + 2] - dU * diffRZ / diffR;

        F[3 * j] = F[3 * j] + dU * diffRX / diffR;
        F[3 * j + 1] = F[3 * j + 1] + dU * diffRY / diffR;
        F[3 * j + 2] = F[3 * j + 2] + dU * diffRZ / diffR;
      }
    }
  }
  return;
}

} // namespace rgpot
