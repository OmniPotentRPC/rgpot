#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Cleri-Rosato Gupta many-body potential.
 *
 * Cleri and Rosato, Phys. Rev. B 48, 22 (1993), doi:10.1103/PhysRevB.48.22:
 *
 *   E = sum_i [ sum_{j != i} A * exp(-p * (r_ij / r_0 - 1))
 *               - sqrt( sum_{j != i} xi^2 * exp(-2 * q * (r_ij / r_0 - 1)) ) ]
 *
 * Each unordered pair contributes 2 A exp(-p (r_ij / r_0 - 1)) to the
 * repulsive sum and xi^2 exp(-2 q (r_ij / r_0 - 1)) to the density rho_i
 * of both ends.
 *
 * With cutoff <= 0 and periodic = false, every pair is included with no
 * cutoff and no minimum-image folding, delegating to the C kernel
 * rgpot_gupta when no stress or per-atom output is requested. A positive
 * cutoff truncates pairs at that distance with no energy shift.
 */

// clang-format off
#include <cstddef>
#include <cstdint>
#include <vector>
// clang-format on
#include "rgpot/Locality.hpp"
#include "rgpot/PairTerms.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/PerAtom.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/Staged.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#if defined(_WIN32)
#define RGPOT_GUPTA_API __declspec(dllexport)
#else
#define RGPOT_GUPTA_API __attribute__((visibility("default")))
#endif

extern "C" {

/// Plain parameter block for @c rgpot_gupta_energy_forces.
/// Field order is part of the ABI.
struct rgpot_gupta_config {
  double A;
  double xi;
  double p;
  double q;
  double r0;
  double cutoff;
  int32_t periodic;
};

/// Energy and forces. Returns 0 on success, 1 on a bad argument.
/// @a box may be null when @c periodic is 0. @a forces has room for
/// @c 3 * natoms doubles. Each thread keeps the potential of its last
/// call: one with the same parameters and atom count reuses its neighbour
/// update and pair terms, and any other builds a new one.
RGPOT_GUPTA_API int
rgpot_gupta_energy_forces(const rgpot_gupta_config *cfg, int32_t natoms,
                          const double *positions, const double *box,
                          double *energy, double *forces);

} // extern "C"

namespace rgpot {

/**
 * @brief Parameters for the Cleri-Rosato Gupta potential.
 *
 * Plain aggregate with user-supplied parameters (A, xi, p, q, r0, cutoff).
 * Defaults are the Cleri-Rosato Zn parameters (F. Cleri and V. Rosato,
 * Phys. Rev. B 48, 22 (1993), doi:10.1103/PhysRevB.48.22; Table 1 of
 * J. P. K. Doye, Phys. Rev. B 68, 195418 (2003), arXiv:cond-mat/0306225),
 * with r0 = 1.0 in reduced length units (r / r0).
 */
struct GuptaConfig {
  double A = 0.1477;  //!< Repulsive prefactor (eV; Cleri-Rosato Zn).
  double xi = 0.8900; //!< Cohesive hopping integral (eV; Cleri-Rosato Zn).
  double p = 9.689;   //!< Repulsive decay exponent (dimensionless; Cleri-Rosato Zn).
  double q = 4.602;   //!< Cohesive decay exponent (dimensionless; Cleri-Rosato Zn).
  double r0 = 1.0;    //!< Reference nearest-neighbour distance (reduced unit at 1.0, or Angstrom with physical r0).
  /// Truncation distance in the coordinate length unit. Values <= 0 keep every pair.
  double cutoff = 0.0;
  /// Minimum image convention. Cluster calls leave this false.
  bool periodic = false;
  /// Write the Cauchy stress sigma = (1/V) dE/dstrain of stress.hpp when the
  /// cell volume is positive.
  bool stress = false;
};

/**
 * @class GuptaPot
 * @brief Cleri-Rosato Gupta energy and analytic forces.
 * @ingroup rgpot_potentials
 */
class GuptaPot : public Potential<GuptaPot> {
public:
  GuptaPot() : GuptaPot(GuptaConfig{}) {}

  explicit GuptaPot(const GuptaConfig &c);

  [[nodiscard]] const GuptaConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance,
            .periodic = m_config.periodic,
            .stress = m_config.stress};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Both the repulsive term and the density stop at the cutoff. A cutoff <= 0
  /// keeps every pair, so there is no finite range. The staged evaluation
  /// exchanges rho.
  [[nodiscard]] Locality locality() const noexcept {
    return {.cutoff = m_config.cutoff > 0.0 ? m_config.cutoff : 0.0,
            .densities = 1};
  }

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// forceImpl plus per-atom terms: each atom's embedding -sqrt(rho_i),
  /// and each pair energy and virial split half to each atom.
  void forcePerAtom(const ForceInput &in, ForceOut *out,
                    const PerAtomOut &atoms) const;

  /// Staged.hpp: adds xi^2 * exp(-2 * q * (r / r0 - 1)) to the density of
  /// each atom of a pair.
  void densityStage(const HalfList &list, double *densities) const;

  /// Staged.hpp: -sqrt(rho_i) of each owned atom and the pair forces with
  /// the embedding weights of both atoms, tallied as forcePerAtom tallies them.
  void forceStage(const HalfList &list, const double *densities, double *forces,
                  StageTally *tally) const;

  /// Stored Verlet list or a streaming cell walk.
  void setNeighbourMode(nlist::Mode mode) noexcept { mode_ = mode; }

  /// Pairs, over all threads, whose density-pass terms the force pass
  /// reads back. It computes the rest again, with the same bits. The
  /// default keeps 32 MiB of terms.
  void setPairTermLimit(std::size_t pairs) noexcept { term_limit_ = pairs; }

private:
  static constexpr uint64_t kKernelVersion = 1;
  using Terms = PairTerms<3>;

  void compute(const ForceInput &in, ForceOut *out,
               const PerAtomOut *atoms) const;

  GuptaConfig m_config;
  uint64_t m_paramsKey{0};
  nlist::Mode mode_ = nlist::Mode::Auto;
  std::size_t term_limit_ = Terms::kDefaultLimit;
  mutable nlist::Neighbours neighbours_;
  mutable std::vector<Terms> terms_;
  mutable std::vector<double> rho_;
  mutable std::vector<double> weight_;
  mutable std::vector<unsigned char> slot_;
  mutable std::vector<double> grad_;
};

} // namespace rgpot
