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

struct ForceAdd {
  int at;
  double v;
  bool sub;
};

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
  // One thread writes the force array in walk order. Several threads
  // record that same order in private buffers, one contiguous atom range
  // each, and the replay walks those buffers from the first range.
  // The buffers stay allocated for the process. They are not
  // thread-local, because the OpenMP workers write the caller's copy.
  static std::vector<std::vector<ForceAdd>> thread_force;
  static std::vector<std::vector<double>> thread_energy;
  static std::vector<int> force_n;
  static std::vector<int> energy_n;
  double energy_direct = 0.0;
  if (nt != 1) {
    if (static_cast<int>(thread_force.size()) < nt) {
      thread_force.resize(static_cast<std::size_t>(nt));
      thread_energy.resize(static_cast<std::size_t>(nt));
    }
    if (static_cast<int>(force_n.size()) < nt) {
      force_n.resize(static_cast<std::size_t>(nt));
      energy_n.resize(static_cast<std::size_t>(nt));
    }
    const auto &list_off = neighbours_.offsets();
    for (int t = 0; t < nt; ++t) {
      const long begin = (N * t) / nt;
      const long end = (N * (t + 1)) / nt;
      int force_cap = 0;
      int energy_cap = 0;
      for (long i = begin; i < end; ++i) {
        const int span = static_cast<int>(list_off[static_cast<std::size_t>(i) + 1] -
                                           list_off[static_cast<std::size_t>(i)]);
        force_cap += span * 9 + 3;
        energy_cap += span + 1;
      }
      auto &fops = thread_force[static_cast<std::size_t>(t)];
      auto &eops = thread_energy[static_cast<std::size_t>(t)];
      if (static_cast<int>(fops.size()) < force_cap) {
        fops.resize(static_cast<std::size_t>(force_cap));
      }
      if (static_cast<int>(eops.size()) < energy_cap) {
        eops.resize(static_cast<std::size_t>(energy_cap));
      }
      force_n[static_cast<std::size_t>(t)] = 0;
      energy_n[static_cast<std::size_t>(t)] = 0;
    }
  }
  auto emitF = [&](ForceAdd *dst, int &n, int at, double v, bool sub) {
    if (dst == nullptr) {
      if (sub) {
        F[at] -= v;
      } else {
        F[at] += v;
      }
    } else {
      dst[n++] = ForceAdd{at, v, sub};
    }
  };
  auto emitE = [&](double *dst, int &n, double term) {
    if (dst == nullptr) {
      energy_direct += term;
    } else {
      dst[n++] = term;
    }
  };
  auto run = [&](long begin, long end, ForceAdd *fops, int &nf, double *eops, int &ne) {
    std::vector<Delta> deltas;
    deltas.reserve(128);
    const auto &off = neighbours_.offsets();
    const auto &nbr = neighbours_.neighbours();
    for (long i = begin; i < end; ++i) {
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
        // r == 0 still contributes the pair energy. The direct loop does
        // not drop that pair, and 1/r there is the same division.
        if (r2 > cutoff2) {
          continue;
        }
        const auto slotp = static_cast<std::size_t>(p);
        const bool pure = atomicNrs[j] == epar.Z && slotp < reuse_ok_.size() &&
                          rev_.size() == reuse_ok_.size();
        // One thread walks chunks in order, so a lower atom has already
        // stored this pair. Several threads only trust a store from the
        // same chunk. The stored value is the lower atom's evaluation,
        // which matches a fresh one when both atoms share coefficients.
        const bool ready = nt == 1 ? j < i : (j >= begin && j < i);
        double rho_term = 0.0;
        double fscale = 0.0;
        double r = 0.0;
        double invR = 0.0;
        bool reused = false;
        if (r2 > 0.0 && pure && ready && reuse_ok_[slotp] != 0 && reuse_r2_[slotp] == r2) {
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
          if (pure && j > i && (nt == 1 || j < end)) {
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
          emitF(fops, nf, static_cast<int>(3 * i), fx, false);
          emitF(fops, nf, static_cast<int>(3 * i + 1), fy, false);
          emitF(fops, nf, static_cast<int>(3 * i + 2), fz, false);
          emitF(fops, nf, static_cast<int>(3 * j), fx, true);
          emitF(fops, nf, static_cast<int>(3 * j + 1), fy, true);
          emitF(fops, nf, static_cast<int>(3 * j + 2), fz, true);
          emitE(eops, ne, phi_r);
        }
      }
      emitE(eops, ne, embedding_function(epar.func_coeff, dens));
      const double dF = embedding_force(epar.func_coeff, dens);
      std::size_t slot = 0;
      for (; slot < deltas.size() && deltas[slot].atom < i; ++slot) {
        const Delta &d = deltas[slot];
        emitF(fops, nf, static_cast<int>(3 * d.atom), dF * d.x, true);
        emitF(fops, nf, static_cast<int>(3 * d.atom + 1), dF * d.y, true);
        emitF(fops, nf, static_cast<int>(3 * d.atom + 2), dF * d.z, true);
      }
      emitF(fops, nf, static_cast<int>(3 * i), dF * dix, true);
      emitF(fops, nf, static_cast<int>(3 * i + 1), dF * diy, true);
      emitF(fops, nf, static_cast<int>(3 * i + 2), dF * diz, true);
      for (; slot < deltas.size(); ++slot) {
        const Delta &d = deltas[slot];
        emitF(fops, nf, static_cast<int>(3 * d.atom), dF * d.x, true);
        emitF(fops, nf, static_cast<int>(3 * d.atom + 1), dF * d.y, true);
        emitF(fops, nf, static_cast<int>(3 * d.atom + 2), dF * d.z, true);
      }
    }
  };
  if (nt == 1) {
    for (int c = 0; c < k; ++c) {
      const auto range = rgpot::nlist::partition(N, c, k);
      int ignored = 0;
      int ignored_e = 0;
      run(range.begin, range.end, nullptr, ignored, nullptr, ignored_e);
    }
    *U = energy_direct;
    return;
  }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    const long begin = (N * thread) / nt;
    const long end = (N * (thread + 1)) / nt;
    auto &fops = thread_force[static_cast<std::size_t>(thread)];
    auto &eops = thread_energy[static_cast<std::size_t>(thread)];
    int nf = 0;
    int ne = 0;
    run(begin, end, fops.data(), nf, eops.data(), ne);
    force_n[static_cast<std::size_t>(thread)] = nf;
    energy_n[static_cast<std::size_t>(thread)] = ne;
  }
  double energy = 0.0;
  for (int t = 0; t < nt; ++t) {
    const double *eterms = thread_energy[static_cast<std::size_t>(t)].data();
    const int nterm = energy_n[static_cast<std::size_t>(t)];
    for (int s = 0; s < nterm; ++s) {
      energy += eterms[s];
    }
    const ForceAdd *adds = thread_force[static_cast<std::size_t>(t)].data();
    const int nadd = force_n[static_cast<std::size_t>(t)];
    for (int s = 0; s < nadd; ++s) {
      if (adds[s].sub) {
        F[adds[s].at] -= adds[s].v;
      } else {
        F[adds[s].at] += adds[s].v;
      }
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
