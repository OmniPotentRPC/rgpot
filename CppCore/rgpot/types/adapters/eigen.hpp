#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Conversion utilities between Eigen and native types.
 *
 * This file contains inline adapter functions for integrating the Eigen
 * linear algebra library with the native @c AtomMatrix and @c std::vector
 * types used in the rgpot library.
 */

// clang-format off
#include <Eigen/Dense>
// clang-format on
#include <array>
#include <vector>

#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace rgpot {
namespace types {
namespace adapt {
namespace eigen {

/**
 * @brief Converts an Eigen matrix to a native AtomMatrix.
 * @param matrix  The source Eigen matrix.
 * @return An @c AtomMatrix instance with copied data.
 */
inline AtomMatrix convertToAtomMatrix(const Eigen::MatrixXd &matrix) {
  AtomMatrix result(matrix.rows(), matrix.cols());
  for (int i = 0; i < matrix.rows(); ++i) {
    for (int j = 0; j < matrix.cols(); ++j) {
      result(i, j) = matrix(i, j);
    }
  }
  return result;
}

/**
 * @brief Converts a native AtomMatrix to an Eigen matrix.
 * @param atomMatrix  The source native matrix.
 * @return An @c Eigen::MatrixXd mapped to the original data.
 * @note This uses @c Eigen::Map for zero-copy access where possible.
 */
inline Eigen::MatrixXd convertToEigen(const AtomMatrix &atomMatrix) {
  return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>(
      atomMatrix.data(), atomMatrix.rows(), atomMatrix.cols());
}

/**
 * @brief Converts an Eigen vector to a standard vector.
 * @param vector  The source Eigen vector.
 * @return A @c std::vector containing the data.
 */
template <typename T>
std::vector<T> convertToVector(const Eigen::VectorX<T> &vector) {
  return std::vector<T>(vector.data(), vector.data() + vector.size());
}

/**
 * @brief Converts a 3x3 Eigen matrix to a nested standard array.
 * @param matrix  The 3x3 Eigen matrix.
 * @return A @c std::array of arrays representing the matrix.
 */
inline std::array<std::array<double, 3>, 3>
convertToEigen3d(const Eigen::Matrix3d &matrix) {
  std::array<std::array<double, 3>, 3> result;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      result[i][j] = matrix(i, j);
    }
  }
  return result;
}

} // namespace eigen
} // namespace adapt
} // namespace types
} // namespace rgpot
