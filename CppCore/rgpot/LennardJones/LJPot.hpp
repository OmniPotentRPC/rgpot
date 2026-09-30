#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Lennard-Jones potential class.
 *
 * This file defines the @c LJPot class, which implements a standard  12-6
 * Lennard-Jones potential with a shifted cutoff for use in  atomic simulations.
 */

// clang-format off
#include <cmath>
#include <utility>
#include <vector>
#include <stdexcept>
// clang-format on
#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the shifted 12-6 Lennard-Jones potential.
 *
 * Plain aggregate with inline defaults: the parameter convention for
 * every classical potential (construct with designated initializers).
 */
struct LJConfig {
  double u0 = 1.0;      //!< Well depth (eV).
  double cutoff = 15.0; //!< Truncation distance (Angstrom).
  double psi = 1.0;     //!< Zero-crossing distance (Angstrom).
};

/**
 * @class LJPot
 * @brief Implementation of a shifted 12-6 Lennard-Jones potential.
 * @ingroup rgpot_potentials
 */
class LJPot : public Potential<LJPot> {
public:
  LJPot() : LJPot(LJConfig{}) {}

  explicit LJPot(const LJConfig &c)
      : Potential(PotType::LJ),
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

  [[nodiscard]] const LJConfig &config() const noexcept { return m_config; }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.stress = true};
  }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
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
  static constexpr uint64_t kKernelVersion = 1;

  double u0;       //!< Well depth parameter.
  double cuttOffR; //!< Distance beyond which potential is truncated.
  double psi;      //!< Distance at which the inter-particle potential is zero.
  double cuttOffU; //!< Potential energy value at the cutoff distance.
  LJConfig m_config;
  uint64_t m_paramsKey{0};
};

} // namespace rgpot
