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
#include <array>
#include <cmath>
#include <limits>
#include <vector>
// clang-format on

#include "rgpot/LennardJones/LJPair.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PairEval.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {

/**
 * @class LJPot
 * @details
 *
 * Pairwise interactions within the cutoff radius, minimum image convention
 * on an orthogonal or sheared box. Pairs come from the shared neighbour
 * layer: a cell list, a Verlet skin, and either a stored list or a
 * streaming walk. The pair term runs during that walk. A streaming call
 * does not copy a half-list. A stored call keeps candidates out to the
 * cutoff plus the skin and drops the rest from the current distances, so
 * a later call inside half the skin reuses the candidates. Several
 * slices split the atoms, each with a private force buffer, and the
 * buffers are added in slice order. A nonzero switch width uses the quintic
 * switch. Cauchy stress is published from the kernel's strain accumulator.
 *
 * @warning The box is assumed to be orthogonal.
 *
 */
void LJPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void LJPot::forcePerAtom(const ForceInput &in, ForceOut *out, const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void LJPot::compute(const ForceInput &in, ForceOut *out, const PerAtomOut *atoms) const {
  const long N = in.nAtoms;
  const double *R = in.pos;
  const double *box = in.box;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(box);
  *U = 0;
  for (long i = 0; i < N; i++) {
    F[3 * i] = 0;
    F[3 * i + 1] = 0;
    F[3 * i + 2] = 0;
  }
  if (N < 2 || cuttOffR <= 0.0) {
    publishCauchyStress(out, strain, volume);
    return;
  }

  const simd::LjParams params =
      makeLjParams(u0, psi, cuttOffU, m_config.switch_width, m_switch);
  const auto n_atoms = static_cast<std::size_t>(N);
  // One angstrom of skin. A stored list keeps pairs out to cutoff + skin
  // and the kernel below drops those past the cutoff. Stream asks for no
  // skin: the bins are not reused unless the coordinates match.
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : 1.0;
  const std::array<bool, 3> periodic{{true, true, true}};
  nlist::updateCells(neighbours_, R, n_atoms, box, cuttOffR, skin, nlist::Image::Round,
                     mode_, periodic);
  const double cutoff2 = cuttOffR * cuttOffR;
  runCellTeamPairs(
      N, neighbours_, F, *U, strain, atoms,
      [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
          double r2, double *force, double &energy, double *virial,
          const PerAtomOut *share) {
        if (!(i < j) || r2 > cutoff2) {
          return;
        }
        // Two different atoms on the same point. Skipping the pair reports a
        // finite energy of zero and hides the singularity.
        if (!(r2 > 0.0)) {
          energy = std::numeric_limits<double>::quiet_NaN();
          return;
        }
        double pair_e = 0.0;
        double fscale = 0.0;
        evalLjPair(r2, params, pair_e, fscale);
        energy += pair_e;
        scatterPairForce(force, i, j, pair_e, fscale, dx, dy, dz, virial,
                         share);
      });
  publishCauchyStress(out, strain, volume);
}

void LJPot::forceStage(const HalfList &list, const double *, double *forces,
                       StageTally *tally) const {
  if (cuttOffR <= 0.0) {
    return;
  }
  const simd::LjParams params =
      makeLjParams(u0, psi, cuttOffU, m_config.switch_width, m_switch);
  const double cutoff2 = cuttOffR * cuttOffR;
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(
        list, [&](int i, int j, double dx, double dy, double dz, double r2, bool both) {
          if (r2 > cutoff2) {
            return;
          }
          if (!(r2 > 0.0)) {
            if constexpr (decltype(tallied)::value) {
              tally->energy = std::numeric_limits<double>::quiet_NaN();
            }
            return;
          }
          double pair_e = 0.0;
          double fscale = 0.0;
          evalLjPair(r2, params, pair_e, fscale);
          staged::addPair(tallied, forces, tally, i, j, both, pair_e, fscale, dx, dy, dz);
        });
  });
}

} // namespace rgpot
