#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Definition of the native AtomMatrix class.
 *
 * This file defines a lightweight, row-major matrix class designed for storing
 * atomic coordinates and forces.
 */

// clang-format off
#include <cxxabi.h>
// clang-format on

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

namespace rgpot {
namespace types {

/**
 * @typedef AtomVector
 * @brief Standard vector for atomic data.
 */
using AtomVector = std::vector<double>;

/**
 * @class AtomMatrix
 * @brief A lightweight row-major matrix class for atomic data.
 */
class AtomMatrix {
public:
  /**
   * @brief Default constructor.
   */
  AtomMatrix() : m_rows(0), m_cols(0) {}

  /**
   * @brief Constructor for list initialization.
   * @param list  The nested initializer list.
   */
  AtomMatrix(std::initializer_list<std::initializer_list<double>> list)
      : m_rows(list.size()), m_cols((list.begin())->size()),
        m_data(m_rows * m_cols) {
    size_t rowIdx = 0;
    for (const auto &rowList : list) {
      std::copy(rowList.begin(), rowList.end(),
                m_data.begin() + rowIdx * m_cols);
      ++rowIdx;
    }
  }

  /**
   * @brief Constructor for a matrix of given dimensions.
   * @param rows  Number of rows.
   * @param cols  Number of columns.
   */
  AtomMatrix(size_t rows, size_t cols)
      : m_rows(rows), m_cols(cols), m_data(rows * cols) {}

  /**
   * @brief Creates a matrix initialized with zeroes.
   * @param rows  Number of rows.
   * @param cols  Number of columns.
   * @return A zero-initialized @c AtomMatrix.
   */
  static AtomMatrix Zero(size_t rows, size_t cols) {
    AtomMatrix matrix(rows, cols);
    std::fill(matrix.m_data.begin(), matrix.m_data.end(), 0.0);
    return matrix;
  }

  /**
   * @brief Access element for mutation.
   * @param row  Row index.
   * @param col  Column index.
   * @return Reference to the element.
   */
  double &operator()(size_t row, size_t col) {
    return m_data[row * m_cols + col];
  }

  /**
   * @brief Access element for reading.
   * @param row  Row index.
   * @param col  Column index.
   * @return Const reference to the element.
   */
  const double &operator()(size_t row, size_t col) const {
    return m_data[row * m_cols + col];
  }

  /**
   * @brief Fetches the number of rows.
   * @return Row count.
   */
  size_t rows() const { return m_rows; }

  /**
   * @brief Fetches the number of columns.
   * @return Column count.
   */
  size_t cols() const { return m_cols; }

  /**
   * @brief Fetches the total number of elements.
   * @return Size of the underlying data vector.
   */
  size_t size() const { return m_rows * m_cols; }

  /**
   * @brief Fetches a pointer to the raw data for mutation.
   * @return Raw pointer to memory.
   */
  double *data() { return m_data.data(); }

  /**
   * @brief Fetches a pointer to the raw data for reading.
   * @return Const raw pointer to memory.
   */
  const double *data() const { return m_data.data(); }

  /**
   * @brief Overload for stream insertion.
   * @param os  The output stream.
   * @param matrix  The matrix to print.
   * @return Reference to the output stream.
   */
  friend std::ostream &operator<<(std::ostream &os, const AtomMatrix &matrix) {
    std::ios oldState(nullptr);
    oldState.copyfmt(os);
    os << std::fixed << std::setprecision(5);
    for (size_t i = 0; i < matrix.m_rows; ++i) {
      for (size_t j = 0; j < matrix.m_cols; ++j) {
        double value = matrix(i, j);
        if (std::abs(value) < 0.001) {
          os << std::scientific;
        } else {
          os << std::fixed;
        }
        os << std::setw(12) << value << ' ';
      }
      os << '\n';
    }
    os.copyfmt(oldState);
    return os;
  }

private:
  size_t m_rows; //!< The number of rows in the matrix.
  size_t m_cols; //!< The number of columns in the matrix.
  std::vector<double>
      m_data; //!< The underlying flat container for row-major data.
};

} // namespace types
} // namespace rgpot
