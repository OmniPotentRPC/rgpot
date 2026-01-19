// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <limits>
#include <set>
// clang-format on
#include "rgpot/CuH2/CuH2Pot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
using rgpot::types::AtomMatrix;

namespace rgpot {
void CuH2Pot::forceImpl(const ForceInput &in, ForceOut *out) const {
  std::multiset<double> natmc;
  const auto N = in.nAtoms;
  int natms[2]{0, 0};                // Always Cu, then H
  int ndim{3 * static_cast<int>(N)}; // see main.f90
  for (size_t i = 0; i < N; ++i) {
    natmc.insert(in.atmnrs[i]);
  }
  if (natmc.count(29) <= 0 || natmc.count(1) <= 0) {
    throw std::runtime_error("The system does not have Copper or Hydrogen, but "
                             "the CuH2 potential was requested");
  }
  natms[0] = natmc.count(29); // Cu
  natms[1] = natmc.count(1);  // H
  if (natms[0] + natms[1] != N) {
    throw std::runtime_error("The system has other atom types, but the CuH2 "
                             "potential was requested");
  }

  // The box only takes the diagonal (assumes cubic)
  double box_eam[]{in.box[0], in.box[4], in.box[8]};
  c_force_eam(natms, ndim, box_eam, const_cast<double *>(in.pos), out->F,
              &out->energy);
}
} // namespace rgpot
