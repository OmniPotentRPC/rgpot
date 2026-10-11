/*
** This file is part of eOn.
**
** SPDX-License-Identifier: BSD-3-Clause
**
** Copyright (c) 2010--present, eOn Development Team
** All rights reserved.
**
** Repo:
** https://github.com/TheochemUI/eOn
*/
#include "EAM.h"
#include "Parameters.h"

#include "rgpot/simd/api.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

void EAM::cleanMemory() {
  // RAII: vectors clean themselves. Nothing to do.
}

void EAM::force(long N, const double *R, const int *atomicNrs, double *F,
                double *U, double *variance, const double *fullbox) {
  variance = nullptr;
  std::array<double, 3> box = {fullbox[0], fullbox[4], fullbox[8]};

  std::array<long, 3> num_axis;
  std::array<long, 3> cell_length;

  for (long i = 0; i < 3; i++) {
    if (!(box[i] > 0.0) || !(rc_[i] > 0.0)) {
      throw std::invalid_argument(
          "EAM::force: box diagonal and cutoff must be positive");
    }
    num_axis[i] = static_cast<long>(box[i] / rc_[i]) + 1;
  }
  for (long i = 0; i < 3; i++) {
    cell_length[i] = static_cast<long>(box[i] / (num_axis[i] - 1));
  }

  long num_cells = num_axis[0] * num_axis[1] * num_axis[2];

  if (!initialized_) {
    celllist_old_.assign(num_cells * (N + 1), 0);
    celllist_new_.assign(num_cells * (N + 1), 0);
    neigh_list_.assign(N * (N + 1), 0);
  }

  *U = 0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }

  // Find minimum coordinates for shifting all positions to positive
  double xmin = std::numeric_limits<double>::max();
  double ymin = xmin, zmin = xmin;

  std::vector<double> Rtemp(3 * N);
  std::vector<double> Rnew(3 * N);

  for (long i = 0; i < 3 * N; i += 3) {
    Rtemp[i] = R[i];
    Rtemp[i + 1] = R[i + 1];
    Rtemp[i + 2] = R[i + 2];
    xmin = std::min(xmin, R[i]);
    ymin = std::min(ymin, R[i + 1]);
    zmin = std::min(zmin, R[i + 2]);
  }
  // Only shift if coordinates are negative
  if (xmin > 0)
    xmin = 0;
  if (ymin > 0)
    ymin = 0;
  if (zmin > 0)
    zmin = 0;

  for (long i = 0; i < 3 * N; i += 3) {
    Rtemp[i] += std::abs(xmin);
    Rtemp[i + 1] += std::abs(ymin);
    Rtemp[i + 2] += std::abs(zmin);
  }

  // Enforce periodic boundary conditions
  for (long i = 0; i < 3 * N; i++) {
    while (Rtemp[i] > box[i % 3]) {
      Rtemp[i] -= box[i % 3];
    }
  }

  std::copy(Rtemp.begin(), Rtemp.end(), Rnew.begin());

  if (!initialized_) {
    new_celllist(N, box.data(), num_axis.data(), cell_length.data(),
                 celllist_new_.data(), num_cells, Rnew.data());
    cell_to_neighbor(N, num_cells, num_axis.data(), cell_length.data(),
                     celllist_new_.data(), neigh_list_.data());
  } else {
    if (update_cell_list(N, num_cells, num_axis.data(), cell_length.data(),
                         celllist_old_.data(), Rnew.data()) > 0) {
      new_celllist(N, box.data(), num_axis.data(), cell_length.data(),
                   celllist_new_.data(), num_cells, Rnew.data());
      cell_to_neighbor(N, num_cells, num_axis.data(), cell_length.data(),
                       celllist_new_.data(), neigh_list_.data());
    }
  }

  calc_force(N, Rnew.data(), atomicNrs, F, U, box.data());

  std::copy(celllist_new_.begin(), celllist_new_.end(), celllist_old_.begin());

  initialized_ = true;
}

EAM::element_parameters EAM::get_element_parameters(int atomic_number) {
  for (int i = 0; i < NPARAMS; i++) {
    if (el_params[i].Z == atomic_number) {
      return el_params[i];
    }
  }
  throw 14324; // Element not found
}

void EAM::calc_force(long N, double *R, const int *atomicNrs, double *F,
                     double *U, const double *box) {
  // Analytical Zhou density and Morse pair, not a spline table. The W=8
  // tree and the in-house exp live in rgpot::simd. The cell list above is
  // unchanged and is not the neighbour loop this kernel walks.
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  *U = 0;
  std::vector<rgpot::simd::EamAtom> atoms(static_cast<std::size_t>(N));
  for (long i = 0; i < N; i++) {
    const element_parameters epar = get_element_parameters(atomicNrs[i]);
    rgpot::simd::EamAtom &atom = atoms[static_cast<std::size_t>(i)];
    atom.beta1 = epar.beta1;
    atom.beta2 = epar.beta2;
    atom.alpha_m = epar.alphaM;
    atom.r_m = epar.Rm;
    atom.d_m = epar.Dm;
    atom.cutoff2 = epar.r_cut * epar.r_cut;
    for (int c = 0; c < 9; ++c) {
      atom.coeff[c] = epar.func_coeff[c];
    }
  }
  const double lengths[3] = {box[0], box[1], box[2]};
  rgpot::simd::eam_force(N, R, atoms.data(), lengths, F, U);
}

void EAM::new_celllist(long N, const double * /*box*/, long *num_axis,
                       long *cell_length, long *celllist_new, long num_cells,
                       double *Rnew) {
  std::vector<long> cell_list(num_cells * (N + 1), -1);
  // Last slot of each cell holds atom count
  for (long i = 0; i < num_cells; i++) {
    cell_list[i * (N + 1) + N] = 0;
  }

  for (long i = 0; i < N; i++) {
    long cx = static_cast<long>(Rnew[3 * i] / cell_length[0]);
    long cy = static_cast<long>(Rnew[3 * i + 1] / cell_length[1]);
    long cz = static_cast<long>(Rnew[3 * i + 2] / cell_length[2]);
    long cell = cx * num_axis[1] * num_axis[2] + cy * num_axis[2] + cz;

    cell_list[cell * (N + 1) + cell_list[cell * (N + 1) + N]] = i;
    cell_list[cell * (N + 1) + N]++;
  }

  std::copy(cell_list.begin(), cell_list.end(), celllist_new);
}

void EAM::cell_to_neighbor(long N, long /*num_of_cells*/, long *num_axis,
                           long * /*cell_length*/, long *celllist_new,
                           long *neigh_list) {
  long num_cells = num_axis[0] * num_axis[1] * num_axis[2];

  for (long i = 0; i < N * (N + 1); i++) {
    neigh_list[i] = -1;
  }

  std::vector<long> neighbors(N + 1);
  std::vector<long> cell_list_copy(num_cells * (N + 1));

  for (long j = 0; j < num_axis[0]; j++) {
    for (long j1 = 0; j1 < num_axis[1]; j1++) {
      for (long j2 = 0; j2 < num_axis[2]; j2++) {
        std::fill(neighbors.begin(), neighbors.end(), -1);

        long cur_index = j * num_axis[1] * num_axis[2] + j1 * num_axis[2] + j2;

        std::copy(celllist_new, celllist_new + num_cells * (N + 1),
                  cell_list_copy.begin());

        if (cell_list_copy[cur_index * (N + 1) + N] == 0) {
          continue;
        }

        for (long d1 = -1; d1 < 2; d1++) {
          for (long d2 = -1; d2 < 2; d2++) {
            for (long d3 = -1; d3 < 2; d3++) {
              std::array<long, 3> pos = {j - d1, j1 - d2, j2 - d3};
              // Periodic wrapping
              for (long y = 0; y < 3; y++) {
                if (pos[y] < 0)
                  pos[y] = num_axis[y] - 1;
                else if (pos[y] >= num_axis[y])
                  pos[y] = 0;
              }
              long neigh_index = pos[0] * num_axis[1] * num_axis[2] +
                                 pos[1] * num_axis[2] + pos[2];

              long num = cell_list_copy[neigh_index * (N + 1) + N];
              for (long y = 0; y < num; y++) {
                long candidate = cell_list_copy[neigh_index * (N + 1) + y];
                // Check not already in neighbors
                bool already = false;
                for (long k = 0; k <= neighbors[N]; k++) {
                  if (neighbors[k] == candidate) {
                    already = true;
                    break;
                  }
                }
                if (!already) {
                  neighbors[N]++;
                  neighbors[neighbors[N]] = candidate;
                }
              }
            }
          }
        }

        for (long i = 0; i < celllist_new[cur_index * (N + 1) + N]; i++) {
          long cur = celllist_new[cur_index * (N + 1) + i];
          neigh_list[cur * (N + 1) + N] = 0;
          for (long temp = 0; temp <= neighbors[N]; temp++) {
            if (cur != neighbors[temp]) {
              neigh_list[cur * (N + 1) + neigh_list[cur * (N + 1) + N]] =
                  neighbors[temp];
              neigh_list[cur * (N + 1) + N]++;
            }
          }
        }
      }
    }
  }
}

int EAM::update_cell_list(long N, long num_cells, long *num_axis,
                          long *cell_length, long *celllist_old, double *Rnew) {
  int changed = 0;
  std::vector<long> table(N);

  for (long i = 0; i < num_cells; i++) {
    for (long j = 0; j < celllist_old[i * (N + 1) + N]; j++) {
      table[celllist_old[i * (N + 1) + j]] = i;
    }
  }

  for (long i = 0; i < N; i++) {
    long cx = static_cast<long>(Rnew[3 * i] / cell_length[0]);
    long cy = static_cast<long>(Rnew[3 * i + 1] / cell_length[1]);
    long cz = static_cast<long>(Rnew[3 * i + 2] / cell_length[2]);
    long cell = cx * num_axis[1] * num_axis[2] + cy * num_axis[2] + cz;
    if (cell != table[i]) {
      changed++;
    }
  }
  return changed;
}

double EAM::embedding_function(const double *func_coeff, double rho) {
  // Horner's method for 8th order polynomial
  double result = func_coeff[8];
  for (int i = 7; i >= 0; i--) {
    result = result * rho + func_coeff[i];
  }
  return result;
}

double EAM::embedding_force(const double *func_coeff, double rho) {
  // Derivative of embedding function via Horner's method
  double result = func_coeff[8] * 8;
  for (int i = 7; i >= 1; i--) {
    result = result * rho + i * func_coeff[i];
  }
  return -result;
}
