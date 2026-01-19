#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <cxxabi.h>
// clang-format on

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

namespace rgpot {
namespace types {
using AtomVector = std::vector<double>;

// A row major matrix class
class AtomMatrix {
public:
  // Default constructor for type_caster in pybind11
  AtomMatrix() : m_rows(0), m_cols(0) {}
  // Constructor to allow list initialization
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

  AtomMatrix(size_t rows, size_t cols)
      : m_rows(rows), m_cols(cols), m_data(rows * cols) {}

  static AtomMatrix Zero(size_t rows, size_t cols) {
    AtomMatrix matrix(rows, cols);
    std::fill(matrix.m_data.begin(), matrix.m_data.end(), 0.0);
    return matrix;
  }

  double &operator()(size_t row, size_t col) {
    return m_data[row * m_cols + col];
  }

  const double &operator()(size_t row, size_t col) const {
    return m_data[row * m_cols + col];
  }

  size_t rows() const { return m_rows; }
  size_t cols() const { return m_cols; }

  // ADDED: Needed for serialization in PotentialCache
  size_t size() const { return m_rows * m_cols; }

  double *data() { return m_data.data(); }
  const double *data() const { return m_data.data(); }

  // Overload the stream insertion operator for AtomMatrix
  friend std::ostream &operator<<(std::ostream &os, const AtomMatrix &matrix) {
    std::ios oldState(nullptr);
    oldState.copyfmt(os); // Save the current format state of the ostream
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
    os.copyfmt(oldState); // Restore the old format state before returning
    return os;
  }

private:
  size_t m_rows, m_cols;
  std::vector<double> m_data;
};

} // namespace types
} // namespace rgpot
