#include "rgpot/EAM/EAMPot.hpp"

#include "EAM.h"

#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {

struct EAMPot::Impl {
  EAM kernel;
};

EAMPot::EAMPot() : EAMPot(EAMCellConfig{}) {}

EAMPot::EAMPot(const EAMCellConfig &)
    : Potential(PotType::EAMCell), m_impl(new Impl) {
  Fnv1a fp;
  fp.u64(1);
  m_paramsKey = fp.h;
}

EAMPot::~EAMPot() { delete m_impl; }

std::uint64_t EAMPot::listBuilds() const noexcept {
  return m_impl->kernel.listBuilds();
}

std::size_t EAMPot::neighborSlots() const noexcept {
  return m_impl->kernel.neighborSlots();
}

bool EAMPot::listReused() const noexcept { return m_impl->kernel.listReused(); }

std::size_t EAMPot::streamPartners() const noexcept {
  return m_impl->kernel.streamPartners();
}

int EAMPot::threadsUsed() const noexcept {
  return m_impl->kernel.threadsUsed();
}

void EAMPot::setNeighbourMode(nlist::Mode mode) noexcept {
  m_impl->kernel.setNeighbourMode(mode);
}

void EAMPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  double energy = 0.0;
  double variance = 0.0;
  m_impl->kernel.force(static_cast<long>(in.nAtoms), in.pos, in.atmnrs, out->F,
                       &energy, &variance, in.box);
  out->energy = energy;
  out->variance = variance;
  out->has_stress = 0;
}

} // namespace rgpot

void EAM::evaluateThreaded(long N, const int *atomicNrs, double *F, double *U) {
  accumulateOwned(N, atomicNrs, F, U);
}

void EAM::accumulateOwned(long N, const int *atomicNrs, double *F, double *U) {
  // Each atom writes its own slots. The force fold then reads the partner
  // through the reverse slot: lower neighbours, this atom's pair terms,
  // this atom's embedding, then the higher neighbours. That is the order
  // the serial walk adds into this atom, so the bits do not depend on
  // which thread owns the atom. The pair record sits in one struct so the
  // reverse read is one line, and each thread's energy steps are already
  // in atom order so the final sum is one contiguous add per thread.
  const auto nslot = neighbours_.slots();
  struct Term {
    double x;
    double y;
    double z;
    double fscale;
    double morse;
  };
  struct Scratch {
    std::vector<double> dix;
    std::vector<double> diy;
    std::vector<double> diz;
    std::vector<double> df;
    std::vector<int> e_at;
    std::vector<char> live;
    std::vector<double> packed;
    std::vector<int> pack_at;
    std::vector<int> pack_used;
    std::vector<Term> term;
  };
  static Scratch scratch;
  if (static_cast<long>(scratch.df.size()) < N) {
    scratch.dix.resize(static_cast<std::size_t>(N));
    scratch.diy.resize(static_cast<std::size_t>(N));
    scratch.diz.resize(static_cast<std::size_t>(N));
    scratch.df.resize(static_cast<std::size_t>(N));
  }
  if (scratch.live.size() < nslot) {
    scratch.live.resize(nslot);
  }
  if (scratch.term.size() < nslot) {
    scratch.term.resize(nslot);
  }
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  scratch.e_at.resize(static_cast<std::size_t>(N) + 1);
  scratch.e_at[0] = 0;
  for (long i = 0; i < N; ++i) {
    const int deg = static_cast<int>(off[static_cast<std::size_t>(i) + 1] -
                                     off[static_cast<std::size_t>(i)]);
    scratch.e_at[static_cast<std::size_t>(i) + 1] =
        scratch.e_at[static_cast<std::size_t>(i)] + deg + 1;
  }
  const int nt = threads_used_;
  scratch.pack_at.resize(static_cast<std::size_t>(nt) + 1);
  scratch.pack_used.assign(static_cast<std::size_t>(nt), 0);
  scratch.pack_at[0] = 0;
  for (int t = 0; t < nt; ++t) {
    const auto range = rgpot::nlist::partition(N, t, nt);
    const int room = scratch.e_at[static_cast<std::size_t>(range.end)] -
                     scratch.e_at[static_cast<std::size_t>(range.begin)];
    scratch.pack_at[static_cast<std::size_t>(t) + 1] =
        scratch.pack_at[static_cast<std::size_t>(t)] + room;
  }
  if (static_cast<int>(scratch.packed.size()) < scratch.pack_at[static_cast<std::size_t>(nt)]) {
    scratch.packed.resize(static_cast<std::size_t>(scratch.pack_at[static_cast<std::size_t>(nt)]));
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
    const auto range = rgpot::nlist::partition(N, thread, nt);
    double *steps = scratch.packed.data() + scratch.pack_at[static_cast<std::size_t>(thread)];
    int ne = 0;
    for (long i = range.begin; i < range.end; ++i) {
      const element_parameters epar = get_element_parameters(atomicNrs[i]);
      const double cutoff2 = epar.r_cut * epar.r_cut;
      double dens = 0.0;
      double dix = 0.0;
      double diy = 0.0;
      double diz = 0.0;
      const long begin = static_cast<long>(off[static_cast<std::size_t>(i)]);
      const long end = static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
      for (long p = begin; p < end; ++p) {
        const auto slot = static_cast<std::size_t>(p);
        const long j = nbr[slot];
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        geometry(i, j, dx, dy, dz, r2);
        Term &term = scratch.term[slot];
        term.x = dx;
        term.y = dy;
        term.z = dz;
        if (r2 > cutoff2) {
          scratch.live[slot] = 0;
          continue;
        }
        scratch.live[slot] = 1;
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double exp_b1 = std::exp(-epar.beta1 * r);
        const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
        const double rho_pair = exp_b1 + 512.0 * exp_b2;
        const double rho_term = r6 * rho_pair;
        const double r5 = r2 * r2 * r;
        const double mag_force_den =
            6.0 * r5 * rho_pair +
            r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
        const double invR = 1.0 / r;
        const double fscale = -mag_force_den * invR;
        term.fscale = fscale;
        dens += rho_term;
        dix += fscale * dx;
        diy += fscale * dy;
        diz += fscale * dz;
        if (j > i) {
          const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
          const double d = 1.0 - expArg;
          const double phi_r = epar.Dm * d * d - epar.Dm;
          const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
          term.morse = mag_force * invR;
          steps[ne] = phi_r;
          ++ne;
        } else {
          term.morse = 0.0;
        }
      }
      steps[ne] = embedding_function(epar.func_coeff, dens);
      ++ne;
      scratch.dix[static_cast<std::size_t>(i)] = dix;
      scratch.diy[static_cast<std::size_t>(i)] = diy;
      scratch.diz[static_cast<std::size_t>(i)] = diz;
      scratch.df[static_cast<std::size_t>(i)] = embedding_force(epar.func_coeff, dens);
    }
    scratch.pack_used[static_cast<std::size_t>(thread)] = ne;
#if defined(_OPENMP)
#pragma omp barrier
#endif
    for (long i = range.begin; i < range.end; ++i) {
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      const long begin = static_cast<long>(off[static_cast<std::size_t>(i)]);
      const long end = static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
      for (long p = begin; p < end; ++p) {
        const long j = nbr[static_cast<std::size_t>(p)];
        if (j >= i) {
          break;
        }
        if (static_cast<std::size_t>(p) >= rev_.size()) {
          continue;
        }
        const auto rev = rev_[static_cast<std::size_t>(p)];
        if (rev < 0 || scratch.live[static_cast<std::size_t>(rev)] == 0) {
          continue;
        }
        const auto rs = static_cast<std::size_t>(rev);
        const Term &back = scratch.term[rs];
        const double pdx = back.x;
        const double pdy = back.y;
        const double pdz = back.z;
        fx -= back.morse * pdx;
        fy -= back.morse * pdy;
        fz -= back.morse * pdz;
        const double sx = -back.fscale * pdx;
        const double sy = -back.fscale * pdy;
        const double sz = -back.fscale * pdz;
        const double dFj = scratch.df[static_cast<std::size_t>(j)];
        fx -= dFj * sx;
        fy -= dFj * sy;
        fz -= dFj * sz;
      }
      struct Held {
        double x;
        double y;
        double z;
      };
      Held held[128];
      int nh = 0;
      const bool stash = (end - begin) <= 128;
      for (long p = begin; p < end; ++p) {
        const auto slot = static_cast<std::size_t>(p);
        const long j = nbr[slot];
        if (j <= i || scratch.live[slot] == 0) {
          continue;
        }
        const Term &here = scratch.term[slot];
        fx += here.morse * here.x;
        fy += here.morse * here.y;
        fz += here.morse * here.z;
        if (!stash || static_cast<std::size_t>(p) >= rev_.size()) {
          continue;
        }
        const auto rev = rev_[slot];
        if (rev < 0 || scratch.live[static_cast<std::size_t>(rev)] == 0) {
          continue;
        }
        const auto rs = static_cast<std::size_t>(rev);
        const Term &back = scratch.term[rs];
        const double dFj = scratch.df[static_cast<std::size_t>(j)];
        const double sx = -back.fscale * back.x;
        const double sy = -back.fscale * back.y;
        const double sz = -back.fscale * back.z;
        held[nh] = Held{dFj * sx, dFj * sy, dFj * sz};
        ++nh;
      }
      const auto is = static_cast<std::size_t>(i);
      const double dFi = scratch.df[is];
      fx -= dFi * scratch.dix[is];
      fy -= dFi * scratch.diy[is];
      fz -= dFi * scratch.diz[is];
      if (stash) {
        for (int h = 0; h < nh; ++h) {
          fx -= held[h].x;
          fy -= held[h].y;
          fz -= held[h].z;
        }
      } else {
        for (long p = begin; p < end; ++p) {
          const long j = nbr[static_cast<std::size_t>(p)];
          if (j <= i) {
            continue;
          }
          if (static_cast<std::size_t>(p) >= rev_.size()) {
            continue;
          }
          const auto rev = rev_[static_cast<std::size_t>(p)];
          if (rev < 0 || scratch.live[static_cast<std::size_t>(rev)] == 0) {
            continue;
          }
          const auto rs = static_cast<std::size_t>(rev);
          const Term &back = scratch.term[rs];
          const double sx = -back.fscale * back.x;
          const double sy = -back.fscale * back.y;
          const double sz = -back.fscale * back.z;
          const double dFj = scratch.df[static_cast<std::size_t>(j)];
          fx -= dFj * sx;
          fy -= dFj * sy;
          fz -= dFj * sz;
        }
      }
      F[3 * i] = fx;
      F[3 * i + 1] = fy;
      F[3 * i + 2] = fz;
    }
  }
  double total = 0.0;
  for (int t = 0; t < nt; ++t) {
    const double *steps = scratch.packed.data() + scratch.pack_at[static_cast<std::size_t>(t)];
    const int ne = scratch.pack_used[static_cast<std::size_t>(t)];
    for (int k = 0; k < ne; ++k) {
      total += steps[k];
    }
  }
  *U = total;
}

void EAM::prefillPure(long N, const int *atomicNrs) {
  const auto nslot = neighbours_.slots();
  if (morse_.size() != nslot) {
    morse_.assign(nslot, 0.0);
    gdx_.assign(nslot, 0.0);
  }
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  const int nt = threads_used_;
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    const auto range = rgpot::nlist::partition(N, thread, nt);
    for (long i = range.begin; i < range.end; ++i) {
      const element_parameters epar = get_element_parameters(atomicNrs[i]);
      const double cutoff2 = epar.r_cut * epar.r_cut;
      const auto row = static_cast<std::size_t>(i);
      for (std::int32_t p = off[row]; p < off[row + 1]; ++p) {
        const long j = nbr[static_cast<std::size_t>(p)];
        if (j <= i || atomicNrs[j] != epar.Z) {
          continue;
        }
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        geometry(i, j, dx, dy, dz, r2);
        if (!(r2 > 0.0) || r2 > cutoff2) {
          continue;
        }
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double exp_b1 = std::exp(-epar.beta1 * r);
        const double exp_b2 = std::exp(-2.0 * epar.beta2 * r);
        const double rho_pair = exp_b1 + 512.0 * exp_b2;
        const double rho_term = r6 * rho_pair;
        const double r5 = r2 * r2 * r;
        const double mag_force_den =
            6.0 * r5 * rho_pair +
            r6 * (-epar.beta1 * exp_b1 - 1024.0 * epar.beta2 * exp_b2);
        const double invR = 1.0 / r;
        const double fscale = -mag_force_den * invR;
        const double expArg = std::exp(-epar.alphaM * (r - epar.Rm));
        const double d = 1.0 - expArg;
        const double phi_r = epar.Dm * d * d - epar.Dm;
        const double mag_force = 2.0 * epar.alphaM * epar.Dm * d * (d - 1.0);
        const auto slot = static_cast<std::size_t>(p);
        reuse_ok_[slot] = 1;
        reuse_r2_[slot] = r2;
        reuse_rho_[slot] = rho_term;
        reuse_fs_[slot] = fscale;
        morse_[slot] = phi_r;
        gdx_[slot] = mag_force * invR;
        const auto rev = rev_[slot];
        if (rev >= 0) {
          const auto rs = static_cast<std::size_t>(rev);
          reuse_ok_[rs] = 1;
          reuse_r2_[rs] = r2;
          reuse_rho_[rs] = rho_term;
          reuse_fs_[rs] = fscale;
        }
      }
    }
  }
}

void EAM::accumulateFilled(long i, const int *atomicNrs, double *F, double &U,
                           std::vector<Delta> &deltas) {
  // Same additions as accumulate. A pure pair whose exponentials were
  // filled ahead of this walk is consumed instead of computed again.
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
  const long begin = static_cast<long>(off[static_cast<std::size_t>(i)]);
  const long end = static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
  for (long p = begin; p < end; ++p) {
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
    if (r2 > cutoff2) {
      continue;
    }
    const auto slot = static_cast<std::size_t>(p);
    if (j > i && atomicNrs[j] == epar.Z && slot < morse_.size() &&
        reuse_ok_[slot] != 0 && reuse_r2_[slot] == r2) {
      const double rho_term = reuse_rho_[slot];
      const double fscale = reuse_fs_[slot];
      dens += rho_term;
      dix += fscale * dx;
      diy += fscale * dy;
      diz += fscale * dz;
      deltas.push_back(Delta{j, -fscale * dx, -fscale * dy, -fscale * dz});
      const double fcomp_scale = gdx_[slot];
      F[3 * i] += fcomp_scale * dx;
      F[3 * i + 1] += fcomp_scale * dy;
      F[3 * i + 2] += fcomp_scale * dz;
      F[3 * j] -= fcomp_scale * dx;
      F[3 * j + 1] -= fcomp_scale * dy;
      F[3 * j + 2] -= fcomp_scale * dz;
      U += morse_[slot];
      continue;
    }
    double rho_term = 0.0;
    double fscale = 0.0;
    double r = 0.0;
    double invR = 0.0;
    const bool pure = atomicNrs[j] == epar.Z && slot < reuse_ok_.size() &&
                      rev_.size() == reuse_ok_.size();
    bool reused = false;
    if (pure && j < i && reuse_ok_[slot] != 0 && reuse_r2_[slot] == r2) {
      rho_term = reuse_rho_[slot];
      fscale = reuse_fs_[slot];
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
        const auto rev = rev_[slot];
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
  std::size_t at = 0;
  for (; at < deltas.size() && deltas[at].atom < i; ++at) {
    const Delta &d = deltas[at];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
  F[3 * i] -= dF * dix;
  F[3 * i + 1] -= dF * diy;
  F[3 * i + 2] -= dF * diz;
  for (; at < deltas.size(); ++at) {
    const Delta &d = deltas[at];
    F[3 * d.atom] -= dF * d.x;
    F[3 * d.atom + 1] -= dF * d.y;
    F[3 * d.atom + 2] -= dF * d.z;
  }
}
