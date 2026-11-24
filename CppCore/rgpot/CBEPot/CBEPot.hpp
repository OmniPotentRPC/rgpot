#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/GradientJet.hpp"
#include "rgpot/Potential.hpp"

#include <memory>
#include <mutex>

struct potlibc_handle;

namespace rgpot {

/// The CH5 2008 surface (H + CH4) of potlibc, `ch52008`.
///
/// Its analytic gradient, Hessian-vector product and batched gradient jet to
/// order three come from one potlibc surface handle.
class CBEPot final : public Potential<CBEPot>, public GradientJet {
public:
  CBEPot();
  ~CBEPot() override;
  void forceImpl(const ForceInput &in, ForceOut *out) const override;
  void forceBatchImpl(const ForceBatch &batch) const override;
  void hessianVectorImpl(const ForceInput &in, const double *direction,
                         double *product) const override;
  [[nodiscard]] int maxJetOrder() const noexcept override;
  DerivativeReport gradientJet(std::size_t systems, std::size_t nAtoms,
                               const double *positions, const int *atomicNrs,
                               const double *boxes, const double *directions,
                               int order, double *jet) const override;
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::ProcessSerial,
            .batched = true,
            .periodic = false,
            .analyticHvp = true};
  }

private:
  struct HandleDeleter {
    void operator()(potlibc_handle *handle) const noexcept;
  };
  std::unique_ptr<potlibc_handle, HandleDeleter> handle_;
  mutable std::mutex mutex_;
};

} // namespace rgpot
