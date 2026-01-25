#pragma once
// MIT License
// Copyright 2024--present Rohit Goswami <HaoZeke>

/**
 * @file cuh2Utils.hpp
 * @author rgpot Developers
 * @date 2026-01-25
 * @brief Utilities for CuH2 system coordinate manipulation.
 *
 * Created on: 2024-01-01
 *
 * This file provides utility functions for setting up and analyzing
 * CuH2 slab systems, with specific support for xtensor-based workflows.
 */

// clang-format off
#include <utility>
#include <vector>
// clang-format on
#include "rgpot/types/AtomMatrix.hpp"

#ifdef WITH_XTENSOR
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xview.hpp"
#endif

using rgpot::types::AtomMatrix;

namespace rgpot {
namespace cuh2 {
namespace utils {

#ifdef WITH_XTENSOR
namespace xts {

/**
 * @brief Perturbs H atoms based on distances from the Cu slab.
 * @param base_positions  Initial coordinate matrix.
 * @param atmNumVec       Vector of atomic numbers.
 * @param hcu_dist        Desired distance between H midpoint and Cu surface.
 * @param hh_dist         Desired distance between Hydrogen atoms.
 * @return Updated xtensor position matrix.
 */
xt::xtensor<double, 2>
perturb_positions(const xt::xtensor<double, 2> &base_positions,
                  const xt::xtensor<int, 1> &atmNumVec, double hcu_dist,
                  double hh_dist);

/**
 * @brief Calculates HH distance and distance from the Cu slab.
 * @param positions  Current coordinate matrix.
 * @param atmNumVec  Vector of atomic numbers.
 * @return A pair containing (H-H distance, H-Slab distance).
 */
std::pair<double, double>
calculateDistances(const xt::xtensor<double, 2> &positions,
                   const xt::xtensor<int, 1> &atmNumVec);

/**
 * @brief Ensures a vector is normalized within a tolerance.
 * @tparam E          The expression type.
 * @tparam ScalarType The numeric type.
 * @param vector         The vector to normalize.
 * @param is_normalized  Manual override to skip calculation.
 * @param tol            The tolerance for unit length.
 * @return Void.
 */
// TODO(rg): This is duplicated from xts::func !!
template <class E, class ScalarType = double>
void ensure_normalized(E &&vector, bool is_normalized = false,
                       ScalarType tol = static_cast<ScalarType>(1e-6)) {
  if (!is_normalized) {
    auto norm = xt::linalg::norm(vector, 2);
    if (norm == 0.0) {
      throw std::runtime_error(
          "Cannot normalize a vector whose norm is smaller than tol");
    }
    if (std::abs(norm - static_cast<ScalarType>(1.0)) >= tol) {
      vector /= norm;
    }
  }
}

} // namespace xts
#endif

} // namespace utils
} // namespace cuh2
} // namespace rgpot
