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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

long cellsAlong(double length, double edge) {
  long n = static_cast<long>(std::floor(length / edge));
  if (n < 1) {
    n = 1;
  }
  // floor can land one cell too high when length/edge is just under an
  // integer, which would make the cell narrower than the search.
  while (n > 1 && length / static_cast<double>(n) < edge) {
    --n;
  }
  return n;
}

long binAtom(double x, double length, long n) {
  // The wrap leaves a coordinate of exactly `length` in place. That point
  // is the periodic image of the origin.
  if (!(x < length)) {
    x = 0.0;
  }
  if (x < 0.0) {
    x = 0.0;
  }
  long i = static_cast<long>(x / length * static_cast<double>(n));
  if (i < 0) {
    i = 0;
  }
  if (i >= n) {
    i = n - 1;
  }
  return i;
}

long wrapCell(long index, long n) {
  if (index < 0) {
    return index + n;
  }
  if (index >= n) {
    return index - n;
  }
  return index;
}

} // namespace

void EAM::cleanMemory() {
  std::vector<double>().swap(wrapped_);
  std::vector<double>().swap(ref_);
  std::vector<int>().swap(species_);
  std::vector<long>().swap(head_);
  std::vector<long>().swap(next_);
  std::vector<long>().swap(off_);
  std::vector<long>().swap(nbr_);
  std::vector<long>().swap(rev_);
  std::vector<long>().swap(gather_);
  std::vector<unsigned>().swap(stamp_);
  std::vector<Delta>().swap(deltas_);
  std::vector<double>().swap(rho_);
  std::vector<double>().swap(dF_);
  std::vector<double>().swap(fscale_);
  std::vector<double>().swap(thread_energy_);
  built_ = false;
  n_atoms_ = 0;
}

EAM::element_parameters EAM::get_element_parameters(int atomic_number) const {
  for (int i = 0; i < NPARAMS; i++) {
    if (el_params[i].Z == atomic_number) {
      return el_params[i];
    }
  }
  throw 14324;
}

void EAM::prepare(long N, const double *R, const double *box) {
  wrapped_.resize(static_cast<std::size_t>(3 * N));
  double xmin = std::numeric_limits<double>::max();
  double ymin = xmin;
  double zmin = xmin;
  for (long i = 0; i < 3 * N; i += 3) {
    wrapped_[static_cast<std::size_t>(i)] = R[i];
    wrapped_[static_cast<std::size_t>(i + 1)] = R[i + 1];
    wrapped_[static_cast<std::size_t>(i + 2)] = R[i + 2];
    xmin = std::min(xmin, R[i]);
    ymin = std::min(ymin, R[i + 1]);
    zmin = std::min(zmin, R[i + 2]);
  }
  if (xmin > 0) {
    xmin = 0;
  }
  if (ymin > 0) {
    ymin = 0;
  }
  if (zmin > 0) {
    zmin = 0;
  }
  for (long i = 0; i < 3 * N; i += 3) {
    wrapped_[static_cast<std::size_t>(i)] += std::abs(xmin);
    wrapped_[static_cast<std::size_t>(i + 1)] += std::abs(ymin);
    wrapped_[static_cast<std::size_t>(i + 2)] += std::abs(zmin);
  }
  for (long i = 0; i < 3 * N; i++) {
    while (wrapped_[static_cast<std::size_t>(i)] > box[i % 3]) {
      wrapped_[static_cast<std::size_t>(i)] -= box[i % 3];
    }
  }
  box_len_[0] = box[0];
  box_len_[1] = box[1];
  box_len_[2] = box[2];
}

bool EAM::reusable(long N, const double *R, const int *atomicNrs,
                   const double *box) const {
  if (!built_ || N != n_atoms_) {
    return false;
  }
  if (box[0] != box_ref_[0] || box[4] != box_ref_[1] || box[8] != box_ref_[2]) {
    return false;
  }
  if (std::memcmp(species_.data(), atomicNrs,
                  static_cast<std::size_t>(N) * sizeof(int)) != 0) {
    return false;
  }
  const double limit = 0.5 * skin_;
  const double limit2 = limit * limit;
  for (long i = 0; i < N; ++i) {
    const double dx = R[3 * i] - ref_[static_cast<std::size_t>(3 * i)];
    const double dy = R[3 * i + 1] - ref_[static_cast<std::size_t>(3 * i + 1)];
    const double dz = R[3 * i + 2] - ref_[static_cast<std::size_t>(3 * i + 2)];
    if (dx * dx + dy * dy + dz * dz > limit2) {
      return false;
    }
  }
  return true;
}

void EAM::buildList(long N, const int *atomicNrs) {
  ++builds_;
  double widest = 0.0;
  for (int i = 0; i < NPARAMS; i++) {
    widest = std::max(widest, el_params[i].r_cut);
  }
  skin_ = kEamSkinOverCutoff * widest;
  const double edge = widest + skin_;
  for (int axis = 0; axis < 3; ++axis) {
    n_axis_[axis] = cellsAlong(box_len_[axis], edge);
  }
  const long nx = n_axis_[0];
  const long ny = n_axis_[1];
  const long nz = n_axis_[2];
  const long ncells = nx * ny * nz;
  head_.assign(static_cast<std::size_t>(ncells), -1);
  next_.resize(static_cast<std::size_t>(N));
  for (long i = 0; i < N; ++i) {
    const double *r = wrapped_.data() + 3 * i;
    const long ix = binAtom(r[0], box_len_[0], nx);
    const long iy = binAtom(r[1], box_len_[1], ny);
    const long iz = binAtom(r[2], box_len_[2], nz);
    const long cell = (ix * ny + iy) * nz + iz;
    next_[static_cast<std::size_t>(i)] = head_[static_cast<std::size_t>(cell)];
    head_[static_cast<std::size_t>(cell)] = i;
  }

  off_.resize(static_cast<std::size_t>(N) + 1);
  nbr_.clear();
  stamp_.assign(static_cast<std::size_t>(ncells), 0);
  unsigned generation = 0;
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;

  for (long i = 0; i < N; ++i) {
    off_[static_cast<std::size_t>(i)] = static_cast<long>(nbr_.size());
    const element_parameters epar = get_element_parameters(atomicNrs[i]);
    const double search = epar.r_cut + skin_;
    const double search2 = search * search;
    const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
    const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
    const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
    const long ix = binAtom(xi, box_len_[0], nx);
    const long iy = binAtom(yi, box_len_[1], ny);
    const long iz = binAtom(zi, box_len_[2], nz);
    ++generation;
    if (generation == 0) {
      std::fill(stamp_.begin(), stamp_.end(), 0);
      generation = 1;
    }
    gather_.clear();
    for (long ox = -1; ox <= 1; ++ox) {
      for (long oy = -1; oy <= 1; ++oy) {
        for (long oz = -1; oz <= 1; ++oz) {
          const long jx = wrapCell(ix + ox, nx);
          const long jy = wrapCell(iy + oy, ny);
          const long jz = wrapCell(iz + oz, nz);
          const long cell = (jx * ny + jy) * nz + jz;
          if (stamp_[static_cast<std::size_t>(cell)] == generation) {
            continue;
          }
          stamp_[static_cast<std::size_t>(cell)] = generation;
          for (long j = head_[static_cast<std::size_t>(cell)]; j >= 0;
               j = next_[static_cast<std::size_t>(j)]) {
            if (j == i) {
              continue;
            }
            double dx = xi - wrapped_[static_cast<std::size_t>(3 * j)];
            double dy = yi - wrapped_[static_cast<std::size_t>(3 * j + 1)];
            double dz = zi - wrapped_[static_cast<std::size_t>(3 * j + 2)];
            if (dx > half0) {
              dx -= box_len_[0];
            } else if (dx < -half0) {
              dx += box_len_[0];
            }
            if (dy > half1) {
              dy -= box_len_[1];
            } else if (dy < -half1) {
              dy += box_len_[1];
            }
            if (dz > half2) {
              dz -= box_len_[2];
            } else if (dz < -half2) {
              dz += box_len_[2];
            }
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= search2) {
              gather_.push_back(j);
            }
          }
        }
      }
    }
    std::sort(gather_.begin(), gather_.end());
    nbr_.insert(nbr_.end(), gather_.begin(), gather_.end());
  }
  off_[static_cast<std::size_t>(N)] = static_cast<long>(nbr_.size());
  // Slot of i in j's row, so a later call can read j's stored scale
  // without searching. The rows stay sorted for the life of the list.
  rev_.resize(nbr_.size());
  for (long i = 0; i < N; ++i) {
    const long lo = off_[static_cast<std::size_t>(i)];
    const long hi = off_[static_cast<std::size_t>(i) + 1];
    for (long p = lo; p < hi; ++p) {
      const long j = nbr_[static_cast<std::size_t>(p)];
      const long jlo = off_[static_cast<std::size_t>(j)];
      const long jhi = off_[static_cast<std::size_t>(j) + 1];
      const long *row = nbr_.data();
      const long *found = std::lower_bound(row + jlo, row + jhi, i);
      rev_[static_cast<std::size_t>(p)] = static_cast<long>(found - row);
    }
  }
  built_ = true;
  n_atoms_ = N;
}

void EAM::accumulate(long i, const int *atomicNrs, double *F, double &U,
                     std::vector<Delta> &deltas) const {
  deltas.clear();
  const element_parameters epar = get_element_parameters(atomicNrs[i]);
  const double cutoff2 = epar.r_cut * epar.r_cut;
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
  const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
  const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
  const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
  double dens = 0.0;
  double dix = 0.0;
  double diy = 0.0;
  double diz = 0.0;
  const long begin = off_[static_cast<std::size_t>(i)];
  const long end = off_[static_cast<std::size_t>(i) + 1];
  for (long p = begin; p < end; ++p) {
    const long j = nbr_[static_cast<std::size_t>(p)];
    double dx = xi - wrapped_[static_cast<std::size_t>(3 * j)];
    double dy = yi - wrapped_[static_cast<std::size_t>(3 * j + 1)];
    double dz = zi - wrapped_[static_cast<std::size_t>(3 * j + 2)];
    if (dx > half0) {
      dx -= box_len_[0];
    } else if (dx < -half0) {
      dx += box_len_[0];
    }
    if (dy > half1) {
      dy -= box_len_[1];
    } else if (dy < -half1) {
      dy += box_len_[1];
    }
    if (dz > half2) {
      dz -= box_len_[2];
    } else if (dz < -half2) {
      dz += box_len_[2];
    }
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > cutoff2) {
      continue;
    }
    const double r = std::sqrt(r2);
    const double r6 = r2 * r2 * r2;
    const double exp_b1 = std::exp(-epar.beta1 * r);
    const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
    const double rho_pair = exp_b1 + 512.0 * exp_b2;
    dens += r6 * rho_pair;
    const double r5 = r2 * r2 * r;
    const double mag_force_den =
        6.0 * r5 * rho_pair +
        r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
    const double invR = 1.0 / r;
    const double fscale = -mag_force_den * invR;
    dix += fscale * dx;
    diy += fscale * dy;
    diz += fscale * dz;
    deltas.push_back(
        Delta{j, -fscale * dx, -fscale * dy, -fscale * dz});
    if (j > i) {
      const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
      const double d = 1.0 - expArg;
      const double phi_r = epar.Dm * d * d - epar.Dm;
      const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
      const double fcomp_scale = mag_force * invR;
      F[3 * i] -= fcomp_scale * dx;
      F[3 * i + 1] -= fcomp_scale * dy;
      F[3 * i + 2] -= fcomp_scale * dz;
      F[3 * j] += fcomp_scale * dx;
      F[3 * j + 1] += fcomp_scale * dy;
      F[3 * j + 2] += fcomp_scale * dz;
      U += phi_r;
    }
  }
  U += embedding_function(epar.func_coeff, dens);
  const double dF = embedding_force(epar.func_coeff, dens);
  std::size_t slot = 0;
  for (; slot < deltas.size() && deltas[slot].atom < i; ++slot) {
    const Delta &d = deltas[slot];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
  F[3 * i] -= dF * dix;
  F[3 * i + 1] -= dF * diy;
  F[3 * i + 2] -= dF * diz;
  for (; slot < deltas.size(); ++slot) {
    const Delta &d = deltas[slot];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
}

int EAM::threadsRequested() const {
#if defined(_OPENMP)
  const char *env = std::getenv("OMP_NUM_THREADS");
  if (env == nullptr || env[0] == '\0') {
    return 1;
  }
  char *end = nullptr;
  const long parsed = std::strtol(env, &end, 10);
  if (end == env || parsed < 1) {
    return 1;
  }
  if (parsed > std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(parsed);
#else
  return 1;
#endif
}

void EAM::geometry(long i, long j, double &dx, double &dy, double &dz,
                   double &r2) const {
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
  dx = wrapped_[static_cast<std::size_t>(3 * i)] -
       wrapped_[static_cast<std::size_t>(3 * j)];
  dy = wrapped_[static_cast<std::size_t>(3 * i + 1)] -
       wrapped_[static_cast<std::size_t>(3 * j + 1)];
  dz = wrapped_[static_cast<std::size_t>(3 * i + 2)] -
       wrapped_[static_cast<std::size_t>(3 * j + 2)];
  if (dx > half0) {
    dx -= box_len_[0];
  } else if (dx < -half0) {
    dx += box_len_[0];
  }
  if (dy > half1) {
    dy -= box_len_[1];
  } else if (dy < -half1) {
    dy += box_len_[1];
  }
  if (dz > half2) {
    dz -= box_len_[2];
  } else if (dz < -half2) {
    dz += box_len_[2];
  }
  r2 = dx * dx + dy * dy + dz * dz;
}

void EAM::evaluate(long N, const int *atomicNrs, double *F, double *U) {
  const int nt = threadsRequested();
  threads_used_ = nt;
  if (nt == 1) {
    double energy = 0.0;
    for (long i = 0; i < N; ++i) {
      accumulate(i, atomicNrs, F, energy, deltas_);
    }
    *U = energy;
    return;
  }
#if defined(_OPENMP)
  // Densities first, then each atom gathers its own force. The scatter
  // loop above writes every neighbour, so threads would need private
  // copies of the whole force array. The gather writes three components
  // and nothing else.
  rho_.resize(static_cast<std::size_t>(N));
  dF_.resize(static_cast<std::size_t>(N));
  fscale_.resize(nbr_.size());
  thread_energy_.assign(static_cast<std::size_t>(nt), 0.0);
#pragma omp parallel num_threads(nt)
  {
    const int t = omp_get_thread_num();
    const long begin =
        static_cast<long>((static_cast<std::int64_t>(N) * t) / nt);
    const long end =
        static_cast<long>((static_cast<std::int64_t>(N) * (t + 1)) / nt);
    for (long i = begin; i < end; ++i) {
      const element_parameters epar = get_element_parameters(atomicNrs[i]);
      const double cutoff2 = epar.r_cut * epar.r_cut;
      double dens = 0.0;
      const long lo = off_[static_cast<std::size_t>(i)];
      const long hi = off_[static_cast<std::size_t>(i) + 1];
      for (long p = lo; p < hi; ++p) {
        const long j = nbr_[static_cast<std::size_t>(p)];
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        geometry(i, j, dx, dy, dz, r2);
        if (r2 > cutoff2) {
          fscale_[static_cast<std::size_t>(p)] = 0.0;
          continue;
        }
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double exp_b1 = std::exp(-epar.beta1 * r);
        const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
        const double rho_pair = exp_b1 + 512.0 * exp_b2;
        dens += r6 * rho_pair;
        const double r5 = r2 * r2 * r;
        const double mag =
            6.0 * r5 * rho_pair +
            r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
        fscale_[static_cast<std::size_t>(p)] = -mag / r;
      }
      rho_[static_cast<std::size_t>(i)] = dens;
      dF_[static_cast<std::size_t>(i)] =
          embedding_force(epar.func_coeff, dens);
    }
#pragma omp barrier
    double energy = 0.0;
    for (long i = begin; i < end; ++i) {
      const element_parameters ei = get_element_parameters(atomicNrs[i]);
      const double cut_i2 = ei.r_cut * ei.r_cut;
      const double dFi = dF_[static_cast<std::size_t>(i)];
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      const long lo = off_[static_cast<std::size_t>(i)];
      const long hi = off_[static_cast<std::size_t>(i) + 1];
      for (long p = lo; p < hi; ++p) {
        const long j = nbr_[static_cast<std::size_t>(p)];
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        geometry(i, j, dx, dy, dz, r2);
        const element_parameters ej = get_element_parameters(atomicNrs[j]);
        const bool in_i = r2 <= cut_i2;
        const bool in_j = r2 <= ej.r_cut * ej.r_cut;
        if (!in_i && !in_j) {
          continue;
        }
        const double r = std::sqrt(r2);
        const double invR = 1.0 / r;
        if (in_i) {
          const double fscale = fscale_[static_cast<std::size_t>(p)];
          fx -= dFi * fscale * dx;
          fy -= dFi * fscale * dy;
          fz -= dFi * fscale * dz;
          if (j > i) {
            const double expArg = std::exp(-ei.alphaM * (r - ei.Rm));
            const double d = 1.0 - expArg;
            energy += ei.Dm * d * d - ei.Dm;
            const double fcomp =
                2.0 * ei.alphaM * ei.Dm * d * (d - 1.0) * invR;
            fx -= fcomp * dx;
            fy -= fcomp * dy;
            fz -= fcomp * dz;
          }
        }
        if (in_j) {
          const std::size_t slot =
              static_cast<std::size_t>(rev_[static_cast<std::size_t>(p)]);
          const double fscale = fscale_[slot];
          const double dFj = dF_[static_cast<std::size_t>(j)];
          fx -= dFj * fscale * dx;
          fy -= dFj * fscale * dy;
          fz -= dFj * fscale * dz;
          if (j < i) {
            const double expArg = std::exp(-ej.alphaM * (r - ej.Rm));
            const double d = 1.0 - expArg;
            const double fcomp =
                2.0 * ej.alphaM * ej.Dm * d * (d - 1.0) * invR;
            fx -= fcomp * dx;
            fy -= fcomp * dy;
            fz -= fcomp * dz;
          }
        }
      }
      energy += embedding_function(ei.func_coeff,
                                   rho_[static_cast<std::size_t>(i)]);
      F[3 * i] = fx;
      F[3 * i + 1] = fy;
      F[3 * i + 2] = fz;
    }
    thread_energy_[static_cast<std::size_t>(t)] = energy;
  }
  double energy = 0.0;
  for (int t = 0; t < nt; ++t) {
    energy += thread_energy_[static_cast<std::size_t>(t)];
  }
  *U = energy;
#else
  double energy = 0.0;
  for (long i = 0; i < N; ++i) {
    accumulate(i, atomicNrs, F, energy, deltas_);
  }
  *U = energy;
#endif
}

void EAM::force(long N, const double *R, const int *atomicNrs, double *F,
                double *U, double *variance, const double *fullbox) {
  (void)variance;
  const double box[3] = {fullbox[0], fullbox[4], fullbox[8]};
  for (int axis = 0; axis < 3; ++axis) {
    if (!(box[axis] > 0.0)) {
      throw std::invalid_argument(
          "EAM::force: box diagonal and cutoff must be positive");
    }
  }
  *U = 0;
  for (long k = 0; k < 3 * N; k++) {
    F[k] = 0.0;
  }
  if (N < 1) {
    return;
  }
  for (long i = 0; i < N; ++i) {
    (void)get_element_parameters(atomicNrs[i]);
  }
  prepare(N, R, box);
  if (!reusable(N, R, atomicNrs, fullbox)) {
    ref_.assign(R, R + 3 * N);
    species_.assign(atomicNrs, atomicNrs + N);
    box_ref_[0] = fullbox[0];
    box_ref_[1] = fullbox[4];
    box_ref_[2] = fullbox[8];
    buildList(N, atomicNrs);
  }
  evaluate(N, atomicNrs, F, U);
}

double EAM::embedding_function(const double *func_coeff, double rho) {
  double result = func_coeff[8];
  for (int i = 7; i >= 0; i--) {
    result = result * rho + func_coeff[i];
  }
  return result;
}

double EAM::embedding_force(const double *func_coeff, double rho) {
  double result = func_coeff[8] * 8;
  for (int i = 7; i >= 1; i--) {
    result = result * rho + i * func_coeff[i];
  }
  return -result;
}
