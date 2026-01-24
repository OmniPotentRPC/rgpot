#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include <vector>

#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/types/AtomMatrix.hpp"
#include <capnp/list.h>
#include <capnp/message.h>

namespace rgpot {
namespace types {
namespace adapt {
namespace capnp {

// --- Functions to convert from Cap'n Proto Readers to Native Types ---

inline AtomMatrix
convertPositionsFromCapnp(const ::capnp::List<double>::Reader &capnpPos,
                          size_t numAtoms) {
  AtomMatrix nativePositions(numAtoms, 3);
  for (size_t i = 0; i < numAtoms * 3; ++i) {
    nativePositions.data()[i] = capnpPos[i];
  }
  return nativePositions;
}

inline std::vector<int>
convertAtomNumbersFromCapnp(const ::capnp::List<int>::Reader &capnpAtmnrs) {
  std::vector<int> nativeAtomTypes(capnpAtmnrs.size());
  for (size_t i = 0; i < capnpAtmnrs.size(); ++i) {
    nativeAtomTypes[i] = capnpAtmnrs[i];
  }
  return nativeAtomTypes;
}

inline std::array<std::array<double, 3>, 3>
convertBoxMatrixFromCapnp(const ::capnp::List<double>::Reader &capnpBox) {
  std::array<std::array<double, 3>, 3> nativeBoxMatrix;
  for (size_t i = 0; i < 3; ++i) {
    nativeBoxMatrix[i] = {capnpBox[i * 3], capnpBox[i * 3 + 1],
                          capnpBox[i * 3 + 2]};
  }
  return nativeBoxMatrix;
}

// --- Functions to convert from Native Types to Cap'n Proto Builders ---

inline void populatePositionsToCapnp(::capnp::List<double>::Builder &capnpPos,
                                     const AtomMatrix &positions) {
  for (size_t i = 0; i < positions.rows() * positions.cols(); ++i) {
    capnpPos.set(i, positions.data()[i]);
  }
}

inline void populateForcesToCapnp(::capnp::List<double>::Builder &capnpForces,
                                  const AtomMatrix &forces) {
  for (size_t i = 0; i < forces.rows() * forces.cols(); ++i) {
    capnpForces.set(i, forces.data()[i]);
  }
}

inline void populateAtomNumbersToCapnp(::capnp::List<int>::Builder &capnpAtmnrs,
                                       const std::vector<int> &atomNumbers) {
  for (size_t i = 0; i < atomNumbers.size(); ++i) {
    capnpAtmnrs.set(i, atomNumbers[i]);
  }
}

inline void populateBoxMatrixToCapnp(
    ::capnp::List<double>::Builder &capnpBox,
    const std::array<std::array<double, 3>, 3> &boxMatrix) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      capnpBox.set(i * 3 + j, boxMatrix[i][j]);
    }
  }
}

} // namespace capnp
} // namespace adapt
} // namespace types
} // namespace rgpot
