// CppCore/rgpot/PotHelpers.cc
// MIT License
// Copyright 2026--present Rohit Goswami <HaoZeke>
#include "PotHelpers.hpp"
#include <stdexcept>

namespace rgpot {

void zeroForceOut(const size_t &nAtoms, ForceOut *efvd) {
  efvd->energy = 0;
  efvd->variance = 0;
  for (size_t idx{0}; idx < nAtoms; idx++) {
    efvd->F[3 * idx] = 0;
    efvd->F[3 * idx + 1] = 0;
    efvd->F[3 * idx + 2] = 0;
  }
}

void checkParams(const ForceInput &params) {
  // Simple sanity check
  if (params.nAtoms <= 0) {
    throw std::runtime_error("Can't work with zero atoms in force call");
  }
}

} // namespace rgpot
