#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file GradientJet.hpp
 * @brief Optional interface of a potential that returns directional
 * derivatives of its gradient to higher order.
 *
 * A potential implements this next to @c Potential<Derived> when it knows the
 * derivatives of its own gradient along a line, for instance by forward-mode
 * differentiation. The engine adapter discovers it with @c dynamic_cast, so
 * @c PotentialBase and its layout stamp do not change and a potential without
 * the interface reports the capability as absent.
 */

#include <cstddef>
#include <cstdint>

#include "rgpot/Potential.hpp"

namespace rgpot {

/// Roundoff report of one derivative evaluation. When @c tracked, the product
/// of @c operations, the machine epsilon and @c scale is a first-order
/// estimate (not a rigorous bound) of the roundoff of the returned
/// derivatives; @c scale is in the units of the returned jet. For a batch
/// @c operations is the sum and @c scale the largest over the systems.
struct DerivativeReport {
  bool tracked = false;
  std::int64_t operations = 0;
  double scale = 0.0;
};

class GradientJet {
public:
  virtual ~GradientJet() = default;

  /// Highest order @c gradientJet can return; at least 1.
  [[nodiscard]] virtual int maxJetOrder() const noexcept = 0;

  /**
   * Writes the jets of @c systems systems sharing @c nAtoms atoms and
   * @c atomicNrs. @c positions and @c directions hold @c systems*3*nAtoms
   * values and @c boxes @c systems*9. @c jet holds
   * @c order*systems*3*nAtoms values laid out [order][system][3*nAtoms]:
   * block @c k - 1 of a system is the @c k-th derivative with respect to
   * @c t of the gradient of the energy at @c positions + t * direction, at
   * @c t = 0, in eV/Angstrom for a direction in Angstrom. Order 1 is the
   * Hessian-vector product. The caller guarantees
   * @c 1 <= order <= maxJetOrder() and non-overlapping buffers.
   */
  virtual DerivativeReport
  gradientJet(std::size_t systems, std::size_t nAtoms, const double *positions,
              const int *atomicNrs, const double *boxes,
              const double *directions, int order, double *jet) const = 0;
};

} // namespace rgpot
