#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <utility>
#include <vector>
// clang-format on
#include "rgpot/Potential.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

// natms(2), ndim, U(1), R(ndim), F(ndim), box(3)
extern "C" void c_force_eam(int *natms, int ndim, double *box, double *R,
                            double *F, double *U);

namespace rgpot {
class CuH2Pot final : public Potential<CuH2Pot> {
public:
  // Constructor initializes potential type and atom properties
  CuH2Pot() : Potential(PotType::CuH2) {}
  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  // Variables
  // EON compatible function
  void force(long N, const double *R, const int *atomicNrs, double *F,
             double *U, const double *box) const;
};
} // namespace rgpot
