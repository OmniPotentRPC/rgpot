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

namespace rgpot {
class LJPot final : public Potential<LJPot> {
public:
  // Constructor initializes potential type and atom properties
  LJPot() : Potential(PotType::LJ), u0{1.0}, cuttOffR{15.0}, psi{1.0} {}

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  // Variables
  double u0;
  double cuttOffR;
  double psi;
  double cuttOffU;
};
} // namespace rgpot
