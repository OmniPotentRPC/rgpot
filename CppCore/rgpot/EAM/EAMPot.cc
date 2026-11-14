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
  const auto nslot = neighbours_.slots();
  if (reuse_ok_.size() != nslot) {
    reuse_r2_.assign(nslot, 0.0);
    reuse_rho_.assign(nslot, 0.0);
    reuse_fs_.assign(nslot, 0.0);
    reuse_ok_.assign(nslot, 0);
  } else {
    std::fill(reuse_ok_.begin(), reuse_ok_.end(), 0);
  }
  prefillPure(N, atomicNrs);
  accumulateOwned(N, atomicNrs, F, U);
}

void EAM::accumulateOwned(long N, const int *atomicNrs, double *F, double *U) {
  // One neighbour record, written by the atom that walks it. The force
  // fold reads the partner's record through the reverse slot, in the
  // order the serial walk would have added that atom.
  struct Term {
    double dx;
    double dy;
    double dz;
    double fscale;
    double fcomp;
    int morse;
    int live;
  };
  struct Scratch {
    std::vector<Term> terms;
    std::vector<double> dix;
    std::vector<double> diy;
    std::vector<double> diz;
    std::vector<double> df;
    std::vector<double> eterm;
    std::vector<int> e_at;
    std::vector<int> e_used;
  };
  thread_local Scratch scratch;
  auto &terms = scratch.terms;
  auto &dixs = scratch.dix;
  auto &diys = scratch.diy;
  auto &dizs = scratch.diz;
  auto &df = scratch.df;
  auto &eterm = scratch.eterm;
  auto &e_at = scratch.e_at;
  auto &e_used = scratch.e_used;
  const auto nslot = neighbours_.slots();
  if (terms.size() < nslot) {
    terms.resize(nslot);
  }
  if (static_cast<long>(df.size()) < N) {
    dixs.resize(static_cast<std::size_t>(N));
    diys.resize(static_cast<std::size_t>(N));
    dizs.resize(static_cast<std::size_t>(N));
    df.resize(static_cast<std::size_t>(N));
  }
  const auto &off = neighbours_.offsets();
  const auto &nbr = neighbours_.neighbours();
  e_at.resize(static_cast<std::size_t>(N) + 1);
  e_used.assign(static_cast<std::size_t>(N), 0);
  e_at[0] = 0;
  for (long i = 0; i < N; ++i) {
    const int deg = static_cast<int>(off[static_cast<std::size_t>(i) + 1] -
                                     off[static_cast<std::size_t>(i)]);
    e_at[static_cast<std::size_t>(i) + 1] = e_at[static_cast<std::size_t>(i)] + deg + 1;
  }
  if (static_cast<int>(eterm.size()) < e_at[static_cast<std::size_t>(N)]) {
    eterm.resize(static_cast<std::size_t>(e_at[static_cast<std::size_t>(N)]));
  }
  const int nt = threads_used_;
  const double half0 = box_len_[0] * 0.5;
  const double half1 = box_len_[1] * 0.5;
  const double half2 = box_len_[2] * 0.5;
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
      const double xi = wrapped_[static_cast<std::size_t>(3 * i)];
      const double yi = wrapped_[static_cast<std::size_t>(3 * i + 1)];
      const double zi = wrapped_[static_cast<std::size_t>(3 * i + 2)];
      double dens = 0.0;
      double dix = 0.0;
      double diy = 0.0;
      double diz = 0.0;
      double *steps = eterm.data() + e_at[static_cast<std::size_t>(i)];
      int ne = 0;
      const long begin = static_cast<long>(off[static_cast<std::size_t>(i)]);
      const long end = static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
      for (long p = begin; p < end; ++p) {
        Term &term = terms[static_cast<std::size_t>(p)];
        term.live = 0;
        term.morse = 0;
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
        term.live = 1;
        term.dx = dx;
        term.dy = dy;
        term.dz = dz;
        const auto slot = static_cast<std::size_t>(p);
        if (j > i && atomicNrs[j] == epar.Z && slot < morse_.size() &&
            reuse_ok_[slot] != 0 && reuse_r2_[slot] == r2) {
          const double rho_term = reuse_rho_[slot];
          const double fscale = reuse_fs_[slot];
          term.fscale = fscale;
          term.fcomp = gdx_[slot];
          term.morse = 1;
          dens += rho_term;
          dix += fscale * dx;
          diy += fscale * dy;
          diz += fscale * dz;
          steps[ne] = morse_[slot];
          ++ne;
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
        }
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
          term.fcomp = mag_force * invR;
          term.morse = 1;
          steps[ne] = phi_r;
          ++ne;
        }
      }
      steps[ne] = embedding_function(epar.func_coeff, dens);
      ++ne;
      e_used[static_cast<std::size_t>(i)] = ne;
      dixs[static_cast<std::size_t>(i)] = dix;
      diys[static_cast<std::size_t>(i)] = diy;
      dizs[static_cast<std::size_t>(i)] = diz;
      df[static_cast<std::size_t>(i)] = embedding_force(epar.func_coeff, dens);
    }
  }
  double total = 0.0;
  for (long i = 0; i < N; ++i) {
    const double *steps = eterm.data() + e_at[static_cast<std::size_t>(i)];
    const int ne = e_used[static_cast<std::size_t>(i)];
    for (int k = 0; k < ne; ++k) {
      total += steps[k];
    }
  }
  *U = total;
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
      double fx = 0.0;
      double fy = 0.0;
      double fz = 0.0;
      const long begin = static_cast<long>(off[static_cast<std::size_t>(i)]);
      const long end = static_cast<long>(off[static_cast<std::size_t>(i) + 1]);
      for (long p = begin; p < end; ++p) {
        const long j = nbr[static_cast<std::size_t>(p)];
        if (j >= i) {
          continue;
        }
        if (static_cast<std::size_t>(p) >= rev_.size()) {
          continue;
        }
        const auto rev = rev_[static_cast<std::size_t>(p)];
        if (rev < 0) {
          continue;
        }
        const Term &t = terms[static_cast<std::size_t>(rev)];
        if (t.live == 0) {
          continue;
        }
        if (t.morse != 0) {
          fx -= t.fcomp * t.dx;
          fy -= t.fcomp * t.dy;
          fz -= t.fcomp * t.dz;
        }
        const double sx = -t.fscale * t.dx;
        const double sy = -t.fscale * t.dy;
        const double sz = -t.fscale * t.dz;
        const double dFj = df[static_cast<std::size_t>(j)];
        fx -= dFj * sx;
        fy -= dFj * sy;
        fz -= dFj * sz;
      }
      for (long p = begin; p < end; ++p) {
        const Term &own = terms[static_cast<std::size_t>(p)];
        if (own.live == 0 || own.morse == 0) {
          continue;
        }
        fx += own.fcomp * own.dx;
        fy += own.fcomp * own.dy;
        fz += own.fcomp * own.dz;
      }
      const double dFi = df[static_cast<std::size_t>(i)];
      fx -= dFi * dixs[static_cast<std::size_t>(i)];
      fy -= dFi * diys[static_cast<std::size_t>(i)];
      fz -= dFi * dizs[static_cast<std::size_t>(i)];
      for (long p = begin; p < end; ++p) {
        const long j = nbr[static_cast<std::size_t>(p)];
        if (j <= i) {
          continue;
        }
        if (static_cast<std::size_t>(p) >= rev_.size()) {
          continue;
        }
        const auto rev = rev_[static_cast<std::size_t>(p)];
        if (rev < 0) {
          continue;
        }
        const Term &t = terms[static_cast<std::size_t>(rev)];
        if (t.live == 0) {
          continue;
        }
        const double sx = -t.fscale * t.dx;
        const double sy = -t.fscale * t.dy;
        const double sz = -t.fscale * t.dz;
        const double dFj = df[static_cast<std::size_t>(j)];
        fx -= dFj * sx;
        fy -= dFj * sy;
        fz -= dFj * sz;
      }
      F[3 * i] = fx;
      F[3 * i + 1] = fy;
      F[3 * i + 2] = fz;
    }
  }
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
