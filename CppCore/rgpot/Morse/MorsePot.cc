// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the Morse potential methods.
 *
 * Pairwise energy and forces for the Morse potential with a shifted cutoff,
 * including periodic boundary handling by the minimum image convention.
 */

// clang-format off
#include <cmath>
#include <limits>
// clang-format on

#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class MorsePot
 * @details
 *
 * With @f$a = e^{\rho_0 (1 - r/r_0)}@f$ the pair term is
 * @f$\epsilon\, a (a - 2)@f$ and its derivative is
 * @f$2 \epsilon \rho_0 (a - a^2) / r_0@f$, which vanishes at @f$r = r_0@f$
 * where @f$a = 1@f$.
 *
 * @warning The box is assumed to be orthogonal, as for the Lennard-Jones
 * potential in this library.
 */
void MorsePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;

  double diffR{0}, diffRX{0}, diffRY{0}, diffRZ{0}, dU{0}, a{0};
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
        a = exp(rho0 * (1 - diffR / r0));

        *U = *U + epsilon * a * (a - 2) - cuttOffU;

        // dV/dr, which is zero at r = r0 where a = 1.
        dU = 2 * epsilon * rho0 * (a - a * a) / r0;

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
