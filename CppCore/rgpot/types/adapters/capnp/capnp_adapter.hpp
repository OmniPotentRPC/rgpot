#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Conversion utilities between Cap'n Proto and native types.
 *
 * This file contains inline adapter functions designed to facilitate the
 * seamless transfer of data between the Cap'n Proto RPC layer and the internal
 * @c Eigen based @c AtomMatrix and other STL types.
 */

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

/**
 * @brief Converts Cap'n Proto position list to an AtomMatrix.
 * @param capnpPos  The reader for the position list.
 * @param numAtoms  The number of atoms in the system.
 * @return An @c AtomMatrix containing the positions.
 */
inline AtomMatrix
convertPositionsFromCapnp(const ::capnp::List<double>::Reader &capnpPos,
                          size_t numAtoms) {
  AtomMatrix nativePositions(numAtoms, 3);
  for (size_t i = 0; i < numAtoms * 3; ++i) {
    nativePositions.data()[i] = capnpPos[i];
  }
  return nativePositions;
}

/**
 * @brief Converts Cap'n Proto atomic number list to an STL vector.
 * @param capnpAtmnrs  The reader for the atomic numbers list.
 * @return A @c std::vector<int> containing the atomic numbers.
 */
inline std::vector<int>
convertAtomNumbersFromCapnp(const ::capnp::List<int>::Reader &capnpAtmnrs) {
  std::vector<int> nativeAtomTypes(capnpAtmnrs.size());
  for (size_t i = 0; i < capnpAtmnrs.size(); ++i) {
    nativeAtomTypes[i] = capnpAtmnrs[i];
  }
  return nativeAtomTypes;
}

/**
 * @brief Converts Cap'n Proto box list to a native 3x3 array.
 * @param capnpBox  The reader for the box matrix list.
 * @return A nested @c std::array of doubles representing the cell.
 */
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

/**
 * @brief Serializes an AtomMatrix of positions to a Cap'n Proto builder.
 * @param capnpPos   The builder for the position list.
 * @param positions  The source @c AtomMatrix.
 * @return Void.
 */
inline void populatePositionsToCapnp(::capnp::List<double>::Builder &capnpPos,
                                     const AtomMatrix &positions) {
  for (size_t i = 0; i < positions.rows() * positions.cols(); ++i) {
    capnpPos.set(i, positions.data()[i]);
  }
}

/**
 * @brief Serializes an AtomMatrix of forces to a Cap'n Proto builder.
 * @param capnpForces  The builder for the force list.
 * @param forces       The source @c AtomMatrix.
 * @return Void.
 */
inline void populateForcesToCapnp(::capnp::List<double>::Builder &capnpForces,
                                  const AtomMatrix &forces) {
  for (size_t i = 0; i < forces.rows() * forces.cols(); ++i) {
    capnpForces.set(i, forces.data()[i]);
  }
}

/**
 * @brief Serializes an STL vector of atomic numbers to a Cap'n Proto builder.
 * @param capnpAtmnrs   The builder for the atomic numbers list.
 * @param atomNumbers  The source @c std::vector.
 * @return Void.
 */
inline void populateAtomNumbersToCapnp(::capnp::List<int>::Builder &capnpAtmnrs,
                                       const std::vector<int> &atomNumbers) {
  for (size_t i = 0; i < atomNumbers.size(); ++i) {
    capnpAtmnrs.set(i, atomNumbers[i]);
  }
}

/**
 * @brief Serializes a native 3x3 box matrix to a Cap'n Proto builder.
 * @param capnpBox    The builder for the box matrix list.
 * @param boxMatrix  The source nested @c std::array.
 * @return Void.
 */
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
