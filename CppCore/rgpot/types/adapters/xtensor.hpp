#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Conversion utilities between xtensor and native types.
 *
 * This file provides adapters for the @c xtensor library, enabling
 * interoperability between multidimensional arrays and rgpot data structures.
 */

#include <array>
#include <vector>
#include <xtensor/xtensor.hpp>

#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace rgpot {
namespace types {
namespace adapt {
namespace xtensor {

/**
 * @brief Converts an xtensor array to a native AtomMatrix.
 * @param matrix  The source 2D xtensor array.
 * @return An @c AtomMatrix containing the copied data.
 */
inline AtomMatrix convertToAtomMatrix(const xt::xtensor<double, 2> &matrix) {
  AtomMatrix result(matrix.shape(0), matrix.shape(1));
  for (size_t i = 0; i < matrix.shape(0); ++i) {
    for (size_t j = 0; j < matrix.shape(1); ++j) {
      result(i, j) = matrix(i, j);
    }
  }
  return result;
}

/**
 * @brief Converts a native AtomMatrix to an xtensor array.
 * @param atomMatrix  The source native matrix.
 * @return A 2D @c xt::xtensor containing the data.
 */
inline xt::xtensor<double, 2> convertToXtensor(const AtomMatrix &atomMatrix) {
  xt::xtensor<double, 2> result =
      xt::zeros<double>({atomMatrix.rows(), atomMatrix.cols()});
  for (size_t i = 0; i < atomMatrix.rows(); ++i) {
    for (size_t j = 0; j < atomMatrix.cols(); ++j) {
      result(i, j) = atomMatrix(i, j);
    }
  }
  return result;
}

/**
 * @brief Converts a 1D xtensor to a standard vector.
 * @param vector  The source 1D xtensor.
 * @return A @c std::vector containing the data.
 */
template <typename T>
std::vector<T> convertToVector(const xt::xtensor<T, 1> &vector) {
  return std::vector<T>(vector.begin(), vector.end());
}

/**
 * @brief Converts a 3x3 xtensor to a nested standard array.
 * @param matrix  The 3x3 xtensor array.
 * @return A nested @c std::array representing the matrix.
 */
inline std::array<std::array<double, 3>, 3>
convertToArray3x3(const xt::xtensor<double, 2> &matrix) {
  std::array<std::array<double, 3>, 3> result;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      result[i][j] = matrix(i, j);
    }
  }
  return result;
}

} // namespace xtensor
} // namespace adapt
} // namespace types
} // namespace rgpot
