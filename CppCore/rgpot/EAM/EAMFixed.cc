// MIT License
// Copyright 2023--present rgpot developers

#include "EAM.h"
#include "StreamSpline.hpp"

#include <algorithm>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

int clampThreads(int nt) {
  return nt < 1 ? 1 : nt;
}

} // namespace

void EAM::evaluateFixed(long N, const int *atomicNrs, double *F, double *U) {
  const int nt = clampThreads(threads_used_);
  const int k = rgpot::nlist::fixedChunks(N);
  const auto nslot = neighbours_.slots();
  if (reuse_ok_.size() != nslot) {
    reuse_r2_.assign(nslot, 0.0);
    reuse_rho_.assign(nslot, 0.0);
    reuse_fs_.assign(nslot, 0.0);
    reuse_ok_.assign(nslot, 0);
  } else if (!reuse_ok_.empty()) {
    std::fill(reuse_ok_.begin(), reuse_ok_.end(), 0);
  }
  std::vector<double> partial(static_cast<std::size_t>(k), 0.0);
  const std::size_t stride = static_cast<std::size_t>(N) * 3;
  std::vector<double> slab(static_cast<std::size_t>(k) * stride, 0.0);
  double *base = slab.data();
  double *part = partial.data();
  auto run = [&](int c) {
    double *local = base + static_cast<std::size_t>(c) * stride;
    const auto range = rgpot::nlist::partition(N, c, k);
    double chunk_e = 0.0;
    std::vector<Delta> deltas;
    deltas.reserve(128);
    const auto &off = neighbours_.offsets();
    const auto &nbr = neighbours_.neighbours();
    for (long i = range.begin; i < range.end; ++i) {
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
      deltas.clear();
      const auto lo = off[static_cast<std::size_t>(i)];
      const auto hi = off[static_cast<std::size_t>(i) + 1];
      for (std::int32_t p = lo; p < hi; ++p) {
        const long j = nbr[static_cast<std::size_t>(p)];
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
        if (!(r2 > 0.0) || r2 > cutoff2) {
          continue;
        }
        const auto slotp = static_cast<std::size_t>(p);
        const bool pure = atomicNrs[j] == epar.Z && slotp < reuse_ok_.size() &&
                          rev_.size() == reuse_ok_.size();
        // One thread walks chunks in order, so a lower atom has already
        // stored this pair. Several threads only trust a store from the
        // same chunk. The stored value is the lower atom's evaluation,
        // which matches a fresh one when both atoms share coefficients.
        const bool ready = nt == 1 ? j < i : (j >= range.begin && j < i);
        double rho_term = 0.0;
        double fscale = 0.0;
        double r = 0.0;
        double invR = 0.0;
        bool reused = false;
        if (pure && ready && reuse_ok_[slotp] != 0 && reuse_r2_[slotp] == r2) {
          rho_term = reuse_rho_[slotp];
          fscale = reuse_fs_[slotp];
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
          if (pure && j > i && (nt == 1 || j < range.end)) {
            const auto rev = rev_[slotp];
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
        deltas.push_back(Delta{j, -fscale * dx, -fscale * dy, -fscale * dz});
        if (j > i) {
          const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
          const double d = 1.0 - expArg;
          const double phi_r = epar.Dm * d * d - epar.Dm;
          const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
          const double fcomp = mag_force * invR;
          const double fx = fcomp * dx;
          const double fy = fcomp * dy;
          const double fz = fcomp * dz;
          local[3 * i] += fx;
          local[3 * i + 1] += fy;
          local[3 * i + 2] += fz;
          local[3 * j] -= fx;
          local[3 * j + 1] -= fy;
          local[3 * j + 2] -= fz;
          chunk_e += phi_r;
        }
      }
      chunk_e += embedding_function(epar.func_coeff, dens);
      const double dF = embedding_force(epar.func_coeff, dens);
      std::size_t slot = 0;
      for (; slot < deltas.size() && deltas[slot].atom < i; ++slot) {
        const Delta &d = deltas[slot];
        local[3 * d.atom] -= dF * d.x;
        local[3 * d.atom + 1] -= dF * d.y;
        local[3 * d.atom + 2] -= dF * d.z;
      }
      local[3 * i] -= dF * dix;
      local[3 * i + 1] -= dF * diy;
      local[3 * i + 2] -= dF * diz;
      for (; slot < deltas.size(); ++slot) {
        const Delta &d = deltas[slot];
        local[3 * d.atom] -= dF * d.x;
        local[3 * d.atom + 1] -= dF * d.y;
        local[3 * d.atom + 2] -= dF * d.z;
      }
    }
    part[c] = chunk_e;
  };
  if (nt == 1) {
    for (int c = 0; c < k; ++c) {
      run(c);
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      for (int c = thread; c < k; c += nt) {
        run(c);
      }
    }
  }
  double energy = 0.0;
  for (int c = 0; c < k; ++c) {
    energy += part[c];
    const double *local = base + static_cast<std::size_t>(c) * stride;
    for (long a = 0; a < 3 * N; ++a) {
      F[a] += local[a];
    }
  }
  *U = energy;
}

void EAM::evaluateStreamFixed(long N, const int *atomicNrs, double *F, double *U) {
  const int nt = clampThreads(threads_used_);
  const int k = rgpot::nlist::fixedChunks(N);
  std::vector<const rgpot::eam::ElementSpline *> kind(static_cast<std::size_t>(N));
  for (long i = 0; i < N; ++i) {
    kind[static_cast<std::size_t>(i)] = &splineFor(atomicNrs[i]);
  }
  const rgpot::eam::ElementSpline *only = kind.empty() ? nullptr : kind[0];
  bool uniform = true;
  for (const auto *spline : kind) {
    if (spline != only) {
      uniform = false;
      break;
    }
  }
  struct Scratch {
    std::vector<double> rho;
    std::vector<double> force;
    std::vector<double> epair;
    int n = -1;
    int k = -1;
  };
  thread_local Scratch scratch;
  if (scratch.n != static_cast<int>(N) || scratch.k != k) {
    scratch.rho.assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(N), 0.0);
    scratch.force.assign(static_cast<std::size_t>(k) * static_cast<std::size_t>(N) * 3, 0.0);
    scratch.epair.assign(static_cast<std::size_t>(k), 0.0);
    scratch.n = static_cast<int>(N);
    scratch.k = k;
  }
  double *rho_rows = scratch.rho.data();
  double *force_rows = scratch.force.data();
  double *epair = scratch.epair.data();
  const auto atoms = static_cast<std::size_t>(N);
  auto run_density = [&](int c) {
    double *row = rho_rows + static_cast<std::size_t>(c) * atoms;
    for (std::size_t i = 0; i < atoms; ++i) {
      row[i] = 0.0;
    }
    double pair = 0.0;
    const auto range = rgpot::nlist::partition(N, c, k);
    neighbours_.forHalfAtoms(range.begin, range.end,
                             [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
                               if (!(r2 > 0.0)) {
                                 return;
                               }
                               if (uniform && only != nullptr) {
                                 if (r2 > only->cutoff2) {
                                   return;
                                 }
                                 const double r = std::sqrt(r2);
                                 const double rho = only->density(r);
                                 row[i] += rho;
                                 row[j] += rho;
                                 pair += only->pairEnergy(r);
                                 return;
                               }
                               const auto *si = kind[static_cast<std::size_t>(i)];
                               const auto *sj = kind[static_cast<std::size_t>(j)];
                               const double r = std::sqrt(r2);
                               if (r2 <= si->cutoff2) {
                                 row[i] += si->density(r);
                                 pair += si->pairEnergy(r);
                               }
                               if (r2 <= sj->cutoff2) {
                                 row[j] += sj->density(r);
                               }
                             });
    epair[c] = pair;
  };
  if (nt == 1) {
    for (int c = 0; c < k; ++c) {
      run_density(c);
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      for (int c = thread; c < k; c += nt) {
        run_density(c);
      }
    }
  }
  rho_.assign(atoms, 0.0);
  dF_.resize(atoms);
  for (long i = 0; i < N; ++i) {
    double dens = 0.0;
    for (int c = 0; c < k; ++c) {
      dens += rho_rows[static_cast<std::size_t>(c) * atoms + static_cast<std::size_t>(i)];
    }
    rho_[static_cast<std::size_t>(i)] = dens;
  }
  double embedded = 0.0;
  for (long i = 0; i < N; ++i) {
    const auto *spline = kind[static_cast<std::size_t>(i)];
    const double dens = rho_[static_cast<std::size_t>(i)];
    dF_[static_cast<std::size_t>(i)] = embedding_force(spline->embed, dens);
    embedded += embedding_function(spline->embed, dens);
  }
  auto run_force = [&](int c) {
    double *row = force_rows + static_cast<std::size_t>(c) * atoms * 3;
    for (std::size_t i = 0; i < atoms * 3; ++i) {
      row[i] = 0.0;
    }
    const auto range = rgpot::nlist::partition(N, c, k);
    neighbours_.forHalfAtoms(
        range.begin, range.end,
        [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
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
          if (scale == 0.0) {
            return;
          }
          const double fx = scale * dx;
          const double fy = scale * dy;
          const double fz = scale * dz;
          row[3 * i] += fx;
          row[3 * i + 1] += fy;
          row[3 * i + 2] += fz;
          row[3 * j] -= fx;
          row[3 * j + 1] -= fy;
          row[3 * j + 2] -= fz;
        });
  };
  if (nt == 1) {
    for (int c = 0; c < k; ++c) {
      run_force(c);
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      for (int c = thread; c < k; c += nt) {
        run_force(c);
      }
    }
  }
  for (long i = 0; i < N; ++i) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    for (int c = 0; c < k; ++c) {
      const double *row = force_rows + static_cast<std::size_t>(c) * atoms * 3;
      fx += row[3 * i];
      fy += row[3 * i + 1];
      fz += row[3 * i + 2];
    }
    F[3 * i] = fx;
    F[3 * i + 1] = fy;
    F[3 * i + 2] = fz;
  }
  double pair = 0.0;
  for (int c = 0; c < k; ++c) {
    pair += epair[c];
  }
  *U = embedded + pair;
}
