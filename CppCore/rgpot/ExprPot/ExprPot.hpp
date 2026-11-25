#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Named-child PES algebra compiled with OpenMM Lepton.
 *
 * @c ExprPot is a @c Potential. Terms are @c unique_ptr<PotentialBase>
 * only.
 */

#ifndef RGPOT_HAS_EXPR
#error "ExprPot requires meson -Dwith_expr=true (-DRGPOT_HAS_EXPR)"
#endif

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rgpot/Potential.hpp"

namespace rgpot {

/**
 * @class ExprPot
 * @brief Energy @f$E = f(E_0, E_1, \ldots)@f$ over named child potentials.
 * @ingroup rgpot_potentials
 *
 * The constructor parses @p expression with @c Lepton::Parser::parse,
 * compiles the energy once, and compiles
 * @c ParsedExpression::differentiate(name) once per term. Unknown,
 * duplicate, unused, or non-identifier names fail closed before any
 * force call. Forces are the chain rule
 * @f$F = \sum_i (\partial f/\partial E_i)\, F_i@f$.
 */
class ExprPot : public Potential<ExprPot> {
public:
  using Term = std::pair<std::string, std::unique_ptr<PotentialBase>>;

  ExprPot(std::string expression, std::vector<Term> terms);
  ~ExprPot() override;

  ExprPot(const ExprPot &) = delete;
  ExprPot &operator=(const ExprPot &) = delete;
  ExprPot(ExprPot &&other) noexcept;
  ExprPot &operator=(ExprPot &&other) noexcept;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  [[nodiscard]] PotCaps caps() const noexcept override;
  [[nodiscard]] uint64_t paramsKey() const noexcept override;
  [[nodiscard]] const std::string &expression() const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] const PotentialBase *child(std::size_t i) const noexcept;

  /// Analytic @f$\partial f/\partial E_{\mathrm{name}}@f$ from the
  /// construct-time compiled derivative, evaluated at the last child
  /// energies (zero if no force call has run).
  [[nodiscard]] double dEnergyDTerm(const std::string &name) const;

private:
  struct Impl;
  std::unique_ptr<Impl> m_impl;
};

} // namespace rgpot
