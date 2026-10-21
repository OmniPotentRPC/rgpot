#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the free-boundary Lennard-Jones cluster potential.
 *
 * This file defines the @c LJClusterPot class, a 12-6 Lennard-Jones
 * potential evaluated without periodic boundary conditions. The kernel is
 * ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/LJCluster), BSD-3-Clause licensed, copyright the eOn
 * Development Team.
 */

// clang-format off
#include <cmath>
#include <cstdint>
// clang-format on
#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/pot_caps.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the free-boundary shifted 12-6 Lennard-Jones
 * potential.
 *
 * Plain aggregate with inline defaults (reduced units): construct with
 * designated initializers.
 */
struct LJClusterConfig {
  double u0 = 1.0;      //!< Well depth (eV).
  double cutoff = 15.0; //!< Truncation distance (Angstrom).
  double psi = 1.0;     //!< Zero-crossing distance (Angstrom).
};

/**
 * @class LJClusterPot
 * @brief Shifted 12-6 Lennard-Jones for isolated clusters.
 *
 * Same pair kernel as @c LJPot with the minimum image convention switched
 * off, so an input cell never folds pair vectors.
 * @ingroup rgpot_potentials
 */
class LJClusterPot : public Potential<LJClusterPot> {
public:
  LJClusterPot() : LJClusterPot(LJClusterConfig{}) {}

  explicit LJClusterPot(const LJClusterConfig &c)
      : Potential(PotType::LJCluster),
        u0{c.u0},
        cuttOffR{c.cutoff},
        psi{c.psi},
        m_config{c} {
    // Shift so U(cuttOffR) = 0 (standard shifted 12-6 LJ).
    const double a = std::pow(psi / cuttOffR, 6.0);
    cuttOffU = 4.0 * u0 * a * (a - 1.0);
    Fnv1a fp;
    fp.u64(kKernelVersion);
    fp.f64(c.u0);
    fp.f64(c.cutoff);
    fp.f64(c.psi);
    m_paramsKey = fp.h;
  }

  [[nodiscard]] const LJClusterConfig &config() const noexcept {
    return m_config;
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  /// Free boundaries: the kernel ignores the cell and never applies the
  /// minimum image convention.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.periodic = false};
  }

  /**
   * @brief Computes the forces and energy for a given configuration.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  /// Bump when the kernel numerics change, so stale cache entries die.
  static constexpr uint64_t kKernelVersion = 2;

  double u0;       //!< Well depth parameter.
  double cuttOffR; //!< Distance beyond which potential is truncated.
  double psi;      //!< Distance at which the inter-particle potential is zero.
  double cuttOffU; //!< Potential energy value at the cutoff distance.
  LJClusterConfig m_config;
  uint64_t m_paramsKey{0};
};

} // namespace rgpot
