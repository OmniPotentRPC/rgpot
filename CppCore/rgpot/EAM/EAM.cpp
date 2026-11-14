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
#include "rgpot/nlist/NeighbourList.hpp"
#include "Parameters.h"
#include "StreamSpline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef EAM_STANDALONE
EAM::EAM(const eonc::Parameters &params)
    : eonc::Potential(eonc::PotType::EAM_AL, params) {}

EAM::~EAM() = default;
#else
EAM::EAM() = default;
EAM::EAM(const EAM &) = default;
EAM::EAM(EAM &&) noexcept = default;
EAM &EAM::operator=(const EAM &) = default;
EAM &EAM::operator=(EAM &&) noexcept = default;
EAM::~EAM() = default;
#endif

// The bytes of this class as this translation unit compiled them. A caller
// built from another header compares its own sizeof(EAM) with this.
extern "C" std::size_t rgpot_eam_object_bytes() { return sizeof(EAM); }

void EAM::cleanMemory() {
  std::vector<double>().swap(wrapped_);
  std::vector<Delta>().swap(deltas_);
  std::vector<std::int32_t>().swap(js_);
  std::vector<double>().swap(rho_);
  std::vector<double>().swap(dF_);
  std::vector<double>().swap(fscale_);
  std::vector<double>().swap(morse_);
  std::vector<double>().swap(gdx_);
  std::vector<double>().swap(gdy_);
  std::vector<double>().swap(gdz_);
  std::vector<double>().swap(ginv_);
  std::vector<std::int32_t>().swap(rev_);
  std::vector<double>().swap(reuse_r2_);
  std::vector<double>().swap(reuse_rho_);
  std::vector<double>().swap(reuse_fs_);
  std::vector<char>().swap(reuse_ok_);
  std::vector<double>().swap(thread_energy_);
  std::vector<double>().swap(partial_f_);
  std::vector<double>().swap(stream_rho_);
  std::vector<double>().swap(stream_force_);
  parts_.clear();
  part_n_ = -1;
  part_nt_ = -1;
  rev_stamp_ = 0;
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

void EAM::buildList(long N) {
  // Auto, Stored, and Stream are the only values the constructor writes.
  // Anything else means this object was not built by that constructor.
  assert(mode_ == rgpot::nlist::Mode::Auto || mode_ == rgpot::nlist::Mode::Stored ||
         mode_ == rgpot::nlist::Mode::Stream);
  double widest = 0.0;
  for (int i = 0; i < NPARAMS; i++) {
    widest = std::max(widest, el_params[i].r_cut);
  }
  double box9[9] = {box_len_[0], 0, 0, 0, box_len_[1], 0, 0, 0, box_len_[2]};
  // A stream does not reuse bins unless the coordinates match, so the
  // skin only adds pairs the kernel then rejects.
  const double skin = mode_ == rgpot::nlist::Mode::Stream ? 0.0 : kEamSkinOverCutoff * widest;
  rgpot::nlist::updateCells(neighbours_, wrapped_.data(), static_cast<std::size_t>(N), box9,
                            widest, skin, rgpot::nlist::Image::Half, mode_, {true, true, true});
  n_atoms_ = N;
  if (neighbours_.streaming()) {
    rev_.clear();
    rev_stamp_ = 0;
    return;
  }
  if (rev_stamp_ != neighbours_.builds()) {
    ensureRev();
    rev_stamp_ = neighbours_.builds();
  }
}

void EAM::ensureRev() {
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  rev_.assign(nbr.size(), 0);
  for (std::size_t i = 0; i + 1 < off.size(); ++i) {
    for (std::int32_t p = off[i]; p < off[i + 1]; ++p) {
      const auto j = static_cast<std::size_t>(nbr[static_cast<std::size_t>(p)]);
      const auto jlo = off[j];
      const auto jhi = off[j + 1];
      const auto *row = nbr.data();
      const auto *found =
          std::lower_bound(row + jlo, row + jhi, static_cast<std::int32_t>(i));
      if (found == row + jhi || *found != static_cast<std::int32_t>(i)) {
        rev_[static_cast<std::size_t>(p)] = -1;
      } else {
        rev_[static_cast<std::size_t>(p)] =
            static_cast<std::int32_t>(found - row);
      }
    }
  }
}

void EAM::ensureParts(long N, int nt) {
  if (part_n_ == N && part_nt_ == nt &&
      static_cast<int>(parts_.size()) == nt) {
    return;
  }
  parts_.resize(static_cast<std::size_t>(nt));
  for (int t = 0; t < nt; ++t) {
    parts_[static_cast<std::size_t>(t)] = rgpot::nlist::partition(N, t, nt);
  }
  part_n_ = N;
  part_nt_ = nt;
}

void EAM::accumulate(long i, const int *atomicNrs, double *F, double &U,
                     std::vector<Delta> &deltas, const std::int32_t *stream_ids,
                     long stream_n) {
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
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  const bool stream = neighbours_.streaming();
  const long begin = stream ? 0 : static_cast<long>(off[static_cast<std::size_t>(i)]);
  const long end =
      stream ? stream_n
             : static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
  for (long p = begin; p < end; ++p) {
    const long j =
        stream ? stream_ids[static_cast<std::size_t>(p)]
               : nbr[static_cast<std::size_t>(p)];
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
    // A pure-metal pair uses the same density coefficients from either
    // end. The lower atom index computes them and leaves them on the
    // higher atom's slot. The higher atom adds that term only when its
    // squared distance is the same bits, which is the value sqrt and
    // exp would have produced here.
    double rho_term = 0.0;
    double fscale = 0.0;
    double r = 0.0;
    double invR = 0.0;
    const bool pure = !stream && atomicNrs[j] == epar.Z &&
                      static_cast<std::size_t>(p) < reuse_ok_.size() &&
                      rev_.size() == reuse_ok_.size();
    bool reused = false;
    if (pure && j < i && reuse_ok_[static_cast<std::size_t>(p)] != 0 &&
        reuse_r2_[static_cast<std::size_t>(p)] == r2) {
      rho_term = reuse_rho_[static_cast<std::size_t>(p)];
      fscale = reuse_fs_[static_cast<std::size_t>(p)];
      reused = true;
    }
    if (!reused) {
      r = std::sqrt(r2);
      const double r6 = r2 * r2 * r2;
      const double exp_b1 = std::exp(-epar.beta1 * r);
      const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
      const double rho_pair = exp_b1 + 512.0 * exp_b2;
      rho_term = r6 * rho_pair;
      const double r5 = r2 * r2 * r;
      const double mag_force_den =
          6.0 * r5 * rho_pair +
          r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
      invR = 1.0 / r;
      fscale = -mag_force_den * invR;
      if (pure && j > i) {
        const auto rev = rev_[static_cast<std::size_t>(p)];
        if (rev >= 0) {
          const auto rs = static_cast<std::size_t>(rev);
          reuse_ok_[rs] = 1;
          reuse_r2_[rs] = r2;
          reuse_rho_[rs] = rho_term;
          reuse_fs_[rs] = fscale;
        }
      }
    }
    dens += rho_term;
    dix += fscale * dx;
    diy += fscale * dy;
    diz += fscale * dz;
    deltas.push_back(
        Delta{j, -fscale * dx, -fscale * dy, -fscale * dz});
    if (j > i) {
      const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
      const double d = 1.0 - expArg;
      const double phi_r = epar.Dm * d * d - epar.Dm;
      // d(phi)/dr = -mag_force, so the force on i is
      // -(d phi/dr) (r_i - r_j) / r = mag_force (r_i - r_j) / r.
      const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
      const double fcomp_scale = mag_force * invR;
      F[3 * i] += fcomp_scale * dx;
      F[3 * i + 1] += fcomp_scale * dy;
      F[3 * i + 2] += fcomp_scale * dz;
      F[3 * j] -= fcomp_scale * dx;
      F[3 * j + 1] -= fcomp_scale * dy;
      F[3 * j + 2] -= fcomp_scale * dz;
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

int EAM::threadsRequested() const { return rgpot::nlist::threadsRequested(); }

void EAM::geometry(long i, long j, double &dx, double &dy, double &dz,
                   double &r2) const {
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
  const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
  const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
  const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
  dx = xi - wrapped_[static_cast<std::size_t>(3 * j)];
  dy = yi - wrapped_[static_cast<std::size_t>(3 * j + 1)];
  dz = zi - wrapped_[static_cast<std::size_t>(3 * j + 2)];
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

const rgpot::eam::ElementSpline &EAM::splineFor(int Z) {
  static std::vector<rgpot::eam::ElementSpline> tables;
  if (tables.capacity() < 8) {
    tables.reserve(8);
  }
  for (const auto &table : tables) {
    if (table.Z == Z) {
      return table;
    }
  }
  const element_parameters *raw = nullptr;
  for (int i = 0; i < NPARAMS; ++i) {
    if (el_params[i].Z == Z) {
      raw = &el_params[i];
      break;
    }
  }
  if (raw == nullptr) {
    throw 14324;
  }
  tables.push_back(rgpot::eam::makeElementSpline(raw->Z, raw->Dm, raw->alphaM, raw->Rm,
                                                 raw->beta1, raw->beta2, raw->r_cut,
                                                 raw->func_coeff));
  return tables.back();
}

void EAM::evaluateDensityStream(long N, const int *atomicNrs, double *F,
                                 double *U) {
  // One half pair, tabulated. The density walk and the force walk
  // share the cell bins. Each thread keeps its own rows, so the
  // partner atom needs no lock and no pair index.
  js_.clear();
  const int nt = threads_used_;
  std::vector<const rgpot::eam::ElementSpline *> kind(
      static_cast<std::size_t>(N));
  for (long i = 0; i < N; ++i) {
    kind[static_cast<std::size_t>(i)] = &splineFor(atomicNrs[i]);
  }
  const rgpot::eam::ElementSpline *only = kind.empty() ? nullptr : kind[0];
  bool uniform = true;
  for (const auto *k : kind) {
    if (k != only) {
      uniform = false;
      break;
    }
  }
  rho_.assign(static_cast<std::size_t>(N), 0.0);
  dF_.resize(static_cast<std::size_t>(N));
  struct DTerm {
    std::int32_t i;
    std::int32_t j;
    double rho_i;
    double rho_j;
    double epair;
  };
  struct FTerm {
    std::int32_t i;
    std::int32_t j;
    double scale;
    double dx;
    double dy;
    double dz;
  };
  std::vector<std::vector<DTerm>> density_parts(static_cast<std::size_t>(nt));
  double epair = 0.0;
  auto density_of = [&](int t, std::int32_t i, std::int32_t j, double, double, double, double r2) {
    if (!(r2 > 0.0)) {
      return;
    }
    DTerm term{i, j, 0, 0, 0};
    if (uniform && only != nullptr) {
      if (r2 > only->cutoff2) {
        return;
      }
      const double r = std::sqrt(r2);
      const double rho = only->density(r);
      term.rho_i = rho;
      term.rho_j = rho;
      term.epair = only->pairEnergy(r);
    } else {
      const auto *si = kind[static_cast<std::size_t>(i)];
      const auto *sj = kind[static_cast<std::size_t>(j)];
      const double r = std::sqrt(r2);
      if (r2 <= si->cutoff2) {
        term.rho_i = si->density(r);
        term.epair = si->pairEnergy(r);
      }
      if (r2 <= sj->cutoff2) {
        term.rho_j = sj->density(r);
      }
    }
    if (nt == 1) {
      rho_[static_cast<std::size_t>(term.i)] += term.rho_i;
      rho_[static_cast<std::size_t>(term.j)] += term.rho_j;
      epair += term.epair;
    } else {
      density_parts[static_cast<std::size_t>(t)].push_back(term);
    }
  };
  neighbours_.forCellTeam(true, nt, density_of);
  for (const auto &part : density_parts) {
    for (const DTerm &term : part) {
      rho_[static_cast<std::size_t>(term.i)] += term.rho_i;
      rho_[static_cast<std::size_t>(term.j)] += term.rho_j;
      epair += term.epair;
    }
  }
  double embedded = 0.0;
  for (long i = 0; i < N; ++i) {
    const auto *s = kind[static_cast<std::size_t>(i)];
    const double dens = rho_[static_cast<std::size_t>(i)];
    dF_[static_cast<std::size_t>(i)] = embedding_force(s->embed, dens);
    embedded += embedding_function(s->embed, dens);
  }
  std::vector<std::vector<FTerm>> force_parts(static_cast<std::size_t>(nt));
  auto force_of = [&](int t, std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
    if (!(r2 > 0.0)) {
      return;
    }
    const double r = std::sqrt(r2);
    double scale = 0.0;
    if (uniform && only != nullptr) {
      if (r2 > only->cutoff2) {
        return;
      }
      const double fs = only->densityForce(r);
      scale = only->pairForce(r) -
              (dF_[static_cast<std::size_t>(i)] + dF_[static_cast<std::size_t>(j)]) * fs;
    } else {
      const auto *si = kind[static_cast<std::size_t>(i)];
      const auto *sj = kind[static_cast<std::size_t>(j)];
      if (r2 <= si->cutoff2) {
        scale += si->pairForce(r);
        scale -= dF_[static_cast<std::size_t>(i)] * si->densityForce(r);
      }
      if (r2 <= sj->cutoff2) {
        scale -= dF_[static_cast<std::size_t>(j)] * sj->densityForce(r);
      }
    }
    if (scale != 0.0) {
      if (nt == 1) {
        const double fx = scale * dx;
        const double fy = scale * dy;
        const double fz = scale * dz;
        F[3 * i] += fx;
        F[3 * i + 1] += fy;
        F[3 * i + 2] += fz;
        F[3 * j] -= fx;
        F[3 * j + 1] -= fy;
        F[3 * j + 2] -= fz;
      } else {
        force_parts[static_cast<std::size_t>(t)].push_back(FTerm{i, j, scale, dx, dy, dz});
      }
    }
  };
  neighbours_.forCellTeam(true, nt, force_of);
  if (nt > 1) {
    for (long i = 0; i < 3 * N; ++i) {
      F[i] = 0.0;
    }
  }
  for (const auto &part : force_parts) {
    for (const FTerm &term : part) {
      const double fx = term.scale * term.dx;
      const double fy = term.scale * term.dy;
      const double fz = term.scale * term.dz;
      F[3 * term.i] += fx;
      F[3 * term.i + 1] += fy;
      F[3 * term.i + 2] += fz;
      F[3 * term.j] -= fx;
      F[3 * term.j + 1] -= fy;
      F[3 * term.j + 2] -= fz;
    }
  }
  *U = embedded + epair;
}

void EAM::evaluate(long N, const int *atomicNrs, double *F, double *U) {
  const int nt = threadsRequested();
  threads_used_ = nt;
  if (neighbours_.streaming()) {
    evaluateDensityStream(N, atomicNrs, F, U);
    return;
  }
  const auto nslot = neighbours_.slots();
  if (reuse_ok_.size() != nslot) {
    reuse_r2_.assign(nslot, 0.0);
    reuse_rho_.assign(nslot, 0.0);
    reuse_fs_.assign(nslot, 0.0);
    reuse_ok_.assign(nslot, 0);
  } else {
    std::fill(reuse_ok_.begin(), reuse_ok_.end(), 0);
  }
  // Several threads parenthesize the pair sum differently, and a parallel
  // prefill of the exponentials is slower than this walk because the walk
  // still evaluates the pair term. One thread keeps the serial order.
  double energy = 0.0;
  for (long i = 0; i < N; ++i) {
    accumulate(i, atomicNrs, F, energy, deltas_, nullptr, 0);
  }
  *U = energy;
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
  buildList(N);
  // The fill pays for itself from 8 threads, and from 4 threads once the
  // cell holds at least 1024 atoms. Every other call keeps evaluate().
  const int nt = threadsRequested();
  if (!neighbours_.streaming() && nt > 1 && (nt >= 8 || N >= 1024)) {
    threads_used_ = nt;
    evaluateThreaded(N, atomicNrs, F, U);
    return;
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
