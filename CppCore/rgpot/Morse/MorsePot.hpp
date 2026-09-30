#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the Morse potential class.
 *
 * This file defines the @c MorsePot class, which implements the pairwise
 * Morse potential with a shifted cutoff. The kernel is ported from eOn
 * (https://github.com/TheochemUI/eOn, client/potentials/Morse), BSD-3-Clause
 * licensed, copyright the eOn Development Team, 2010.
 */

// clang-format off
#include <cmath>
#include <cstdint>
// clang-format on
#include "rgpot/ParamHash.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"

namespace rgpot {

/**
 * @brief Parameters for the shifted Morse potential.
 *
 * Plain aggregate with inline defaults (platinum): construct with
 * designated initializers.
 */
struct MorseConfig {
  double De = 0.7102;  //!< Well depth (eV).
  double a = 1.6047;   //!< Range parameter (1/Angstrom).
  double re = 2.8970;  //!< Equilibrium pair distance (Angstrom).
  double cutoff = 9.5; //!< Truncation distance (Angstrom).
};

/**
 * @class MorsePot
 * @brief Pairwise Morse potential, @f$V(r) = D_e [1 - e^{-a(r - r_e)}]^2 -
 * D_e@f$, shifted so that @f$V(r_\mathrm{cut}) = 0@f$.
 * @ingroup rgpot_potentials
 */
class MorsePot : public Potential<MorsePot> {
public:
  MorsePot() : MorsePot(MorseConfig{}) {}

  explicit MorsePot(const MorseConfig &c)
      : Potential(PotType::Morse),
        De{c.De},
        a{c.a},
        re{c.re},
        cuttOffR{c.cutoff},
        m_config{c} {
    // Shift so U(cuttOffR) = 0, evaluating the same closed form as the
    // pair kernel below.
    const double d = 1.0 - std::exp(-a * (cuttOffR - re));
    energyCutoff = De * d * d - De;
    Fnv1a fp;
    fp.u64(kKernelVersion);
    fp.f64(c.De);
    fp.f64(c.a);
    fp.f64(c.re);
    fp.f64(c.cutoff);
    m_paramsKey = fp.h;
  }

  [[nodiscard]] const MorseConfig &config() const noexcept { return m_config; }

  /// Energy offset subtracted from every pair term (the unshifted well
  /// depth at the cutoff).
  [[nodiscard]] double energyShift() const noexcept { return energyCutoff; }

  [[nodiscard]] uint64_t paramsKey() const noexcept override {
    return m_paramsKey;
  }

  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.stress = true};
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

  double De;           //!< Well depth parameter.
  double a;            //!< Range parameter.
  double re;           //!< Equilibrium pair distance.
  double cuttOffR;     //!< Distance beyond which the potential is truncated.
  double energyCutoff; //!< Potential energy value at the cutoff distance.
  MorseConfig m_config;
  uint64_t m_paramsKey{0};
};

} // namespace rgpot
