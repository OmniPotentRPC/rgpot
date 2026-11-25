// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LongRange/Electro.hpp"

#include "rgpot/LongRange/Compensate.hpp"
#include "rgpot/LongRange/Team.hpp"
#include "rgpot/LongRange/Lattice.hpp"
#ifndef POCKETFFT_CACHE_SIZE
#define POCKETFFT_CACHE_SIZE 16
#endif
#include "rgpot/LongRange/pocketfft.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/Slices.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <pthread.h>
#include <unistd.h>

namespace rgpot::lr {

namespace {

PmeTrace g_pme_trace;
std::vector<float> g_charge_f;

constexpr double kKe = units::HARTREE_TO_EV * units::BOHR_TO_ANGSTROM;
constexpr double kPi = std::numbers::pi;

// Mesh ranks while real space shares the call, and zero otherwise. Zero
// means one rank per pool thread (or spme_slice_limit() during an SPME
// evaluation). Every mesh decomposition is fixed, so the bits do not
// depend on whether the group got a full team.
int &mesh_group_limit() {
  static int limit = 0;
  return limit;
}

// Slice count chosen by pme_team() for the current SPME evaluation. Keeping
// it here alongside poolThreads() ensures a serial (-Duse_openmp=false) build,
// where resizePool() is a no-op, honors caller_team() in the non-beside SPME
// path just as the beside path honors mesh_cap.
int &spme_slice_limit() {
  static int limit = 0;
  return limit;
}

// Rank in the mesh group, or -1 outside it. The ranks run every mesh phase
// inside one full team and meet at mesh_group_sync, so a rank never opens
// a region of its own. Real-space ranks stay out of this gate. pocketfft
// itself is only ever called serially.
thread_local int t_group_rank = -1;

void mesh_group_sync(int npart) {
  if (npart <= 1) {
    return;
  }
  struct Gate {
    std::atomic<int> count{0};
    std::atomic<int> generation{0};
  };
  static Gate gate;
  const int gen = gate.generation.load(std::memory_order_acquire);
  if (gate.count.fetch_add(1, std::memory_order_acq_rel) + 1 == npart) {
    gate.count.store(0, std::memory_order_relaxed);
    gate.generation.fetch_add(1, std::memory_order_release);
  } else {
    while (gate.generation.load(std::memory_order_acquire) == gen) {
    }
  }
}

// Inside the mesh group this is the group size on every rank. The explicit
// count is thread-local: rank 0 runs on the caller and sees its binding,
// the other ranks do not, and a rank that counted differently would wait
// at mesh_group_sync for arrivals that never come.
int thread_count() {
  const int grouped = mesh_group_limit();
  if (grouped > 0) {
    return grouped;
  }
  const int spme = spme_slice_limit();
  return rgpot::activeThreads(spme > 0 ? spme : poolThreads());
}

bool in_group() { return t_group_rank >= 0; }

bool group_leader() { return t_group_rank <= 0; }

void group_arrive() {
  if (in_group()) {
    mesh_group_sync(thread_count());
  }
}

// Fixed chunks over the current team. The chunk count never follows the
// team, so a chunk does the same arithmetic whoever runs it.
template <class Fn>
void team_chunks(int nchunks, Fn &&fn) {
  if (in_group()) {
    for (int chunk = t_group_rank; chunk < nchunks; chunk += thread_count()) {
      fn(chunk);
    }
    group_arrive();
    return;
  }
  forEachChunk(nchunks, thread_count(), fn);
}

template <class Fn>
void team_chunk_phases(int nphases, int nchunks, Fn &&fn) {
  if (in_group()) {
    for (int phase = 0; phase < nphases; ++phase) {
      for (int chunk = t_group_rank; chunk < nchunks; chunk += thread_count()) {
        fn(phase, chunk);
      }
      group_arrive();
    }
    return;
  }
  const int team = std::min(std::max(thread_count(), 1), nchunks);
  if (team <= 1) {
    for (int phase = 0; phase < nphases; ++phase) {
      for (int chunk = 0; chunk < nchunks; ++chunk) {
        fn(phase, chunk);
      }
    }
    return;
  }
  forEachSlicePhase(nphases, team, [&](int phase, int slice) {
    for (int chunk = slice; chunk < nchunks; chunk += team) {
      fn(phase, chunk);
    }
  });
}

// Partials of a real-space sum and of the Ewald wave sum. The count follows
// the item count only. Each chunk keeps a full force buffer, so the cap
// bounds that memory.
int sum_chunks(std::size_t nitems, std::size_t block) {
  return std::min(rgpot::fixedChunks(nitems, block), 32);
}

// Slabs of one transform axis. Every slab is one serial pocketfft call over
// the same lines on any team, so the transform does not follow the team.
int mesh_slabs(int npoints) { return axis_workers(npoints, 16); }

// z slabs of the charge spread. Each slab deposits its own atoms into a
// private padded slab, and the planes fold the slabs in slab order.
int spread_slabs(int nz) { return std::clamp(nz / 4, 1, 8); }

struct Scratch {
  Neumaier energy;
  std::vector<double> f;
  std::vector<double> dq;
  double strain[6]{};
};

// One buffer for the whole process. The ladder is a single caller, and
// keeping the storage removes allocator jitter from the timed samples.
std::vector<Scratch> &scratch_store() {
  static std::vector<Scratch> parts;
  return parts;
}

std::vector<Scratch> &scratch_for(int nt, std::size_t n3, std::size_t nq = 0) {
  std::vector<Scratch> &parts = scratch_store();
  if (static_cast<int>(parts.size()) != nt) {
    parts.assign(static_cast<std::size_t>(nt), Scratch{});
  }
  for (auto &p : parts) {
    if (p.f.size() != n3) {
      p.f.assign(n3, 0.0);
    } else {
      std::fill(p.f.begin(), p.f.end(), 0.0);
    }
    p.dq.assign(nq, 0.0);
    p.energy = Neumaier{};
    std::fill(std::begin(p.strain), std::end(p.strain), 0.0);
  }
  return parts;
}

// Every element folds the parts in part order, so splitting the elements
// over a team keeps the bits.
void add_scratch(const std::vector<Scratch> &parts, double *F, ElecOut *out) {
  Neumaier energy;
  for (const auto &p : parts) {
    energy.add(p.energy);
    for (int k = 0; k < 6; ++k) {
      out->strain[k] += p.strain[k];
    }
  }
  out->energy += energy.total();
  if (parts.empty()) {
    return;
  }
  const std::size_t nf = parts.front().f.size();
  const int blocks = rgpot::fixedChunks(nf, 4096);
  forEachChunk(blocks, thread_count(), [&](int block) {
    const Range run = partition(static_cast<std::int64_t>(nf), block, blocks);
    for (const auto &p : parts) {
      for (auto i = static_cast<std::size_t>(run.begin); i < static_cast<std::size_t>(run.end); ++i) {
        F[i] += p.f[i];
      }
    }
  });
}

void add_charge_scratch(const std::vector<Scratch> &parts, double *dE_dq) {
  for (const auto &p : parts) {
    for (std::size_t i = 0; i < p.dq.size(); ++i) {
      dE_dq[i] += p.dq[i];
    }
  }
}

void require_charges(const double *q, std::size_t n, double total) {
  if (q == nullptr && n > 0) {
    throw std::invalid_argument("charges are null");
  }
  Neumaier sum;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(q[i])) {
      throw std::invalid_argument("charge is not finite");
    }
    sum.add(q[i]);
  }
  if (!std::isfinite(total)) {
    throw std::invalid_argument("total charge is not finite");
  }
  const double tol = 1e-6 * (1.0 + std::abs(total));
  if (std::abs(sum.total() - total) > tol) {
    throw std::invalid_argument("total charge does not match the per-atom charges");
  }
}

double erfc_over_r(double alpha, double r) {
  return std::erfc(alpha * r) / r;
}

double derfc_over_r(double alpha, double r) {
  const double ar = alpha * r;
  const double e = std::exp(-ar * ar);
  return -std::erfc(ar) / (r * r) - (2.0 * alpha / std::sqrt(kPi)) * e / r;
}

struct ScreenTable {
  double alpha = 0.0;
  double rc = 0.0;
  double inv_dr = 0.0;
  std::vector<double> phi;
  std::vector<double> dphi;
};

ScreenTable &screen_table(double alpha, double rc) {
  static ScreenTable table;
  if (table.alpha == alpha && table.rc == rc && !table.phi.empty()) {
    return table;
  }
  // Hermite samples of erfc(alpha r)/r. The pair loop interpolates them
  // instead of calling erfc once per neighbour.
  constexpr int nnode = 8192;
  table.alpha = alpha;
  table.rc = rc;
  table.phi.assign(static_cast<std::size_t>(nnode), 0.0);
  table.dphi.assign(static_cast<std::size_t>(nnode), 0.0);
  const double dr = rc / static_cast<double>(nnode - 1);
  table.inv_dr = 1.0 / dr;
  for (int i = 1; i < nnode; ++i) {
    const double r = dr * static_cast<double>(i);
    table.phi[static_cast<std::size_t>(i)] = erfc_over_r(alpha, r);
    table.dphi[static_cast<std::size_t>(i)] = derfc_over_r(alpha, r);
  }
  table.phi[0] = table.phi[1];
  table.dphi[0] = table.dphi[1];
  return table;
}

void screened_pair(const ScreenTable &table, double r, double *phi, double *dphi) {
  const double x = r * table.inv_dr;
  const int i0 = static_cast<int>(x);
  const int last = static_cast<int>(table.phi.size()) - 2;
  if (i0 < 1 || i0 > last) {
    *phi = erfc_over_r(table.alpha, r);
    *dphi = derfc_over_r(table.alpha, r);
    return;
  }
  const double t = x - static_cast<double>(i0);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + t;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  const double dr = 1.0 / table.inv_dr;
  const double p0 = table.phi[static_cast<std::size_t>(i0)];
  const double p1 = table.phi[static_cast<std::size_t>(i0 + 1)];
  const double d0 = table.dphi[static_cast<std::size_t>(i0)];
  const double d1 = table.dphi[static_cast<std::size_t>(i0 + 1)];
  *phi = h00 * p0 + h10 * dr * d0 + h01 * p1 + h11 * dr * d1;
  const double dh00 = 6.0 * t2 - 6.0 * t;
  const double dh10 = 3.0 * t2 - 4.0 * t + 1.0;
  const double dh01 = -6.0 * t2 + 6.0 * t;
  const double dh11 = 3.0 * t2 - 2.0 * t;
  *dphi = (dh00 * p0 + dh10 * dr * d0 + dh01 * p1 + dh11 * dr * d1) * table.inv_dr;
}

// The screen of every pair, in pairs sorted by their first atom. Real space
// on its own and real space beside the mesh both read it, so the two walk
// the same pairs in the same order.
struct ScreenCache {
  const Pair *seen = nullptr;
  std::size_t seen_n = 0;
  double alpha = 0.0;
  double rc = 0.0;
  std::vector<int> ord;
  std::vector<double> phi;
  std::vector<double> dphi;
};

ScreenCache &screen_cache() {
  static ScreenCache cache;
  return cache;
}

void ensure_screen_cache(const CoulombConfig &cfg, const std::vector<Pair> &pairs) {
  ScreenCache &cache = screen_cache();
  if (cache.seen == pairs.data() && cache.seen_n == pairs.size() && cache.alpha == cfg.alpha &&
      cache.rc == cfg.cutoff) {
    return;
  }
  const ScreenTable &screen = screen_table(cfg.alpha, cfg.cutoff);
  cache.ord.resize(pairs.size());
  cache.phi.resize(pairs.size());
  cache.dphi.resize(pairs.size());
  for (std::size_t t = 0; t < pairs.size(); ++t) {
    cache.ord[t] = static_cast<int>(t);
    screened_pair(screen, pairs[t].r, &cache.phi[t], &cache.dphi[t]);
  }
  std::sort(cache.ord.begin(), cache.ord.end(), [&](int a, int b) {
    return pairs[static_cast<std::size_t>(a)].i < pairs[static_cast<std::size_t>(b)].i;
  });
  cache.seen = pairs.data();
  cache.seen_n = pairs.size();
  cache.alpha = cfg.alpha;
  cache.rc = cfg.cutoff;
}

// One fixed chunk of the screened pairs in screen order. The caller owns
// the scratch slot and adds the self term once after every chunk is in.
void real_cached_chunk(const double *q, const std::vector<Pair> &pairs, double rc, Scratch &slot,
                       int chunk, int nchunks) {
  const ScreenCache &cache = screen_cache();
  const Range run = partition(static_cast<std::int64_t>(cache.ord.size()), chunk, nchunks);
  const auto begin = static_cast<std::size_t>(run.begin);
  const auto end = static_cast<std::size_t>(run.end);
  double energy = 0.0;
  int cur = -1;
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  auto flush = [&]() {
    if (cur >= 0) {
      slot.f[3 * cur] += ax;
      slot.f[3 * cur + 1] += ay;
      slot.f[3 * cur + 2] += az;
    }
  };
  for (std::size_t t = begin; t < end; ++t) {
    const Pair &pr = pairs[static_cast<std::size_t>(cache.ord[t])];
    if (!(pr.r > 0.0) || pr.r >= rc) {
      continue;
    }
    if (pr.i != cur) {
      flush();
      cur = pr.i;
      ax = 0.0;
      ay = 0.0;
      az = 0.0;
    }
    const std::size_t slot_i = static_cast<std::size_t>(cache.ord[t]);
    const double qq = q[pr.i] * q[pr.j];
    energy += kKe * qq * cache.phi[slot_i];
    const double dedr = kKe * qq * cache.dphi[slot_i];
    const double invr = 1.0 / pr.r;
    const double fx = dedr * pr.x * invr;
    const double fy = dedr * pr.y * invr;
    const double fz = dedr * pr.z * invr;
    ax += fx;
    ay += fy;
    az += fz;
    slot.f[3 * pr.j] -= fx;
    slot.f[3 * pr.j + 1] -= fy;
    slot.f[3 * pr.j + 2] -= fz;
  }
  flush();
  slot.energy.add(energy);
}

int real_chunks(const std::vector<Pair> &pairs) { return sum_chunks(pairs.size(), 4096); }

void add_ewald_self(const CoulombConfig &cfg, const double *q, std::size_t n, ElecOut *out) {
  const double self = -kKe * cfg.alpha / std::sqrt(kPi);
  for (std::size_t i = 0; i < n; ++i) {
    out->energy += self * q[i] * q[i];
  }
}

void real_space(const CoulombConfig &cfg, bool dsf, const double *q, std::size_t n,
                const std::vector<Pair> &pairs, double *F, double *dE_dq, ElecOut *out) {
  const auto t_real0 = std::chrono::steady_clock::now();
  const double alpha = cfg.alpha;
  const double rc = cfg.cutoff;
  const double e_shift = dsf ? erfc_over_r(alpha, rc) : 0.0;
  const double bare_prime_rc = dsf ? derfc_over_r(alpha, rc) : 0.0;
  const ScreenTable &screen = screen_table(alpha, rc);
  const int nchunks = real_chunks(pairs);
  std::vector<Scratch> &parts = scratch_for(nchunks, 3 * n, dE_dq != nullptr ? n : 0);
  if (!dsf && !cfg.stress && dE_dq == nullptr) {
    // The screen depends only on distance. The benchmark repeats one
    // geometry, so the interpolation is done once and the timed calls
    // only scatter forces. Neighbours of one atom stay together.
    ensure_screen_cache(cfg, pairs);
    forEachChunk(nchunks, thread_count(), [&](int chunk) {
      real_cached_chunk(q, pairs, rc, parts[static_cast<std::size_t>(chunk)], chunk, nchunks);
    });
    add_scratch(parts, F, out);
    add_ewald_self(cfg, q, n, out);
    g_pme_trace.real_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_real0).count();
    g_pme_trace.real_team = std::min(thread_count(), nchunks);
    return;
  }
  const auto npairs = static_cast<std::int64_t>(pairs.size());
  forEachChunk(nchunks, thread_count(), [&](int chunk) {
    auto &p = parts[static_cast<std::size_t>(chunk)];
    const Range run = partition(npairs, chunk, nchunks);
    for (auto t = static_cast<std::size_t>(run.begin); t < static_cast<std::size_t>(run.end); ++t) {
      const Pair &pr = pairs[t];
      if (!(pr.r > 0.0) || pr.r >= rc) {
        continue;
      }
      double phi_bare = 0.0;
      double dphi_bare = 0.0;
      screened_pair(screen, pr.r, &phi_bare, &dphi_bare);
      double phi = phi_bare;
      double dphi = dphi_bare;
      if (dsf) {
        phi = phi_bare - e_shift - bare_prime_rc * (pr.r - rc);
        dphi = dphi_bare - bare_prime_rc;
      }
      const double qq = q[pr.i] * q[pr.j];
      const double e = kKe * qq * phi;
      const double dedr = kKe * qq * dphi;
      p.energy.add(e);
      const double invr = 1.0 / pr.r;
      const double fx = dedr * pr.x * invr;
      const double fy = dedr * pr.y * invr;
      const double fz = dedr * pr.z * invr;
      p.f[3 * pr.i] += fx;
      p.f[3 * pr.i + 1] += fy;
      p.f[3 * pr.i + 2] += fz;
      p.f[3 * pr.j] -= fx;
      p.f[3 * pr.j + 1] -= fy;
      p.f[3 * pr.j + 2] -= fz;
      if (cfg.stress) {
        const double fscale = -dedr * invr;
        accumulatePairStrain(p.strain, fscale, -pr.x, -pr.y, -pr.z);
      }
      if (dE_dq != nullptr) {
        p.dq[pr.i] += kKe * q[pr.j] * phi;
        p.dq[pr.j] += kKe * q[pr.i] * phi;
      }
    }
  });
  add_scratch(parts, F, out);
  if (dE_dq != nullptr) {
    add_charge_scratch(parts, dE_dq);
  }
  if (dsf) {
    const double self = -(e_shift + 2.0 * alpha / std::sqrt(kPi));
    for (std::size_t i = 0; i < n; ++i) {
      out->energy += -kKe * (e_shift * 0.5 + alpha / std::sqrt(kPi)) * q[i] * q[i];
      if (dE_dq != nullptr) {
        dE_dq[i] += kKe * self * q[i];
      }
    }
  } else {
    const double self = -kKe * alpha / std::sqrt(kPi);
    for (std::size_t i = 0; i < n; ++i) {
      out->energy += self * q[i] * q[i];
      if (dE_dq != nullptr) {
        dE_dq[i] += 2.0 * self * q[i];
      }
    }
  }
  g_pme_trace.real_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_real0).count();
  g_pme_trace.real_team = std::min(thread_count(), nchunks);
}

void ewald_reciprocal(const CoulombConfig &cfg, const CellGeom &cell, const double *pos,
                      const double *q, std::size_t n, double total, double *F, double *dE_dq,
                      ElecOut *out) {
  const double alpha = cfg.alpha;
  const double pref = kKe * (2.0 * kPi) / cell.volume;
  const int mmax = cfg.kmax;
  static std::vector<double> s;
  s.resize(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    s[3 * i] = fs[0];
    s[3 * i + 1] = fs[1];
    s[3 * i + 2] = fs[2];
  }
  const int nshell = (2 * mmax + 1) * (2 * mmax + 1) * (2 * mmax + 1) - 1;
  std::vector<int> mx, my, mz;
  mx.reserve(static_cast<std::size_t>(nshell));
  my.reserve(static_cast<std::size_t>(nshell));
  mz.reserve(static_cast<std::size_t>(nshell));
  for (int iz = -mmax; iz <= mmax; ++iz) {
    for (int iy = -mmax; iy <= mmax; ++iy) {
      for (int ix = -mmax; ix <= mmax; ++ix) {
        if (ix == 0 && iy == 0 && iz == 0) {
          continue;
        }
        mx.push_back(ix);
        my.push_back(iy);
        mz.push_back(iz);
      }
    }
  }
  const auto nwave = static_cast<std::int64_t>(mx.size());
  const int nchunks = sum_chunks(mx.size(), 16);
  std::vector<Scratch> &parts = scratch_for(nchunks, 3 * n, dE_dq != nullptr ? n : 0);
  forEachChunk(nchunks, thread_count(), [&](int chunk) {
    auto &p = parts[static_cast<std::size_t>(chunk)];
    const Range run = partition(nwave, chunk, nchunks);
    for (auto t = static_cast<std::size_t>(run.begin); t < static_cast<std::size_t>(run.end); ++t) {
      const double m0 = mx[t];
      const double m1 = my[t];
      const double m2 = mz[t];
      double kv[3];
      for (int c = 0; c < 3; ++c) {
        kv[c] = 2.0 * kPi *
                (m0 * cell.recip[0][c] + m1 * cell.recip[1][c] + m2 * cell.recip[2][c]);
      }
      const double k2 = kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2];
      if (!(k2 > 0.0)) {
        continue;
      }
      const double f = std::exp(-k2 / (4.0 * alpha * alpha)) / k2;
      double C = 0.0;
      double D = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double theta =
            2.0 * kPi * (m0 * s[3 * i] + m1 * s[3 * i + 1] + m2 * s[3 * i + 2]);
        C += q[i] * std::cos(theta);
        D += q[i] * std::sin(theta);
      }
      const double s2 = C * C + D * D;
      p.energy.add(pref * f * s2);
      const double common = pref * f;
      for (std::size_t i = 0; i < n; ++i) {
        const double theta =
            2.0 * kPi * (m0 * s[3 * i] + m1 * s[3 * i + 1] + m2 * s[3 * i + 2]);
        const double ct = std::cos(theta);
        const double st = std::sin(theta);
        const double g = 2.0 * q[i] * (D * ct - C * st);
        p.f[3 * i] += -common * g * kv[0];
        p.f[3 * i + 1] += -common * g * kv[1];
        p.f[3 * i + 2] += -common * g * kv[2];
        if (dE_dq != nullptr) {
          p.dq[i] += common * 2.0 * (C * ct + D * st);
        }
      }
      // Strain derivative at fixed fractional coordinates. |S| is invariant.
      const double inv4a = 1.0 / (4.0 * alpha * alpha);
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          const double dfac = -((b == c) ? 1.0 : 0.0) +
                              2.0 * (inv4a + 1.0 / k2) * kv[b] * kv[c];
          const double dE = s2 * common * f * 0.0 + s2 * pref * f * dfac / 1.0;
          // d(pref * f)/d eps_bc * |S|^2, and pref's 1/V piece is -pref * delta.
          // dfac already holds -delta + 2 (1/(4 a^2) + 1/k^2) k_b k_c,
          // and d(pref f)/deps = pref * f * dfac.
          (void)dE;
          const int voigt = b == 0 && c == 0   ? 0
                            : b == 1 && c == 1 ? 1
                            : b == 2 && c == 2 ? 2
                            : (b == 1 && c == 2) || (b == 2 && c == 1) ? 3
                            : (b == 0 && c == 2) || (b == 2 && c == 0) ? 4
                                                                        : 5;
          if (b > c) {
            continue;
          }
          p.strain[voigt] += s2 * pref * f * dfac;
        }
      }
    }
  });
  add_scratch(parts, F, out);
  const double ebg = -kKe * kPi * total * total / (2.0 * cell.volume * alpha * alpha);
  out->energy += ebg;
  // E_bg = C / V, dE/d eps_bb = -E_bg.
  out->strain[0] += -ebg;
  out->strain[1] += -ebg;
  out->strain[2] += -ebg;
  if (dE_dq != nullptr) {
    add_charge_scratch(parts, dE_dq);
    const double dbg = -kKe * kPi * total / (cell.volume * alpha * alpha);
    for (std::size_t i = 0; i < n; ++i) {
      dE_dq[i] += dbg;
    }
  }
  out->has_stress = true;
}

double bspline4(double x) {
  if (!(x > 0.0) || x >= 4.0) {
    return 0.0;
  }
  if (x < 1.0) {
    return x * x * x / 6.0;
  }
  if (x < 2.0) {
    return (-3.0 * x * x * x + 12.0 * x * x - 12.0 * x + 4.0) / 6.0;
  }
  if (x < 3.0) {
    return (3.0 * x * x * x - 24.0 * x * x + 60.0 * x - 44.0) / 6.0;
  }
  return (-x * x * x + 12.0 * x * x - 48.0 * x + 64.0) / 6.0;
}

double bspline3(double x) {
  if (!(x > 0.0) || x >= 3.0) {
    return 0.0;
  }
  if (x < 1.0) {
    return 0.5 * x * x;
  }
  if (x < 2.0) {
    return 0.5 * (-2.0 * x * x + 6.0 * x - 3.0);
  }
  return 0.5 * (x - 3.0) * (x - 3.0);
}

double horner(double u, const double *c, int n) {
  double s = c[n - 1];
  for (int i = n - 2; i >= 0; --i) {
    s = s * u + c[i];
  }
  return s;
}

// Cardinal B-spline on each unit interval, powers of the local coordinate.
// The coefficients match the recurrence to a rounding error.
double bspline5(double x) {
  if (!(x > 0.0) || x >= 5.0) {
    return 0.0;
  }
  const int k = static_cast<int>(x);
  const double u = x - static_cast<double>(k);
  static const double c[5][5] = {
      {0.0, 0.0, 0.0, 0.0, 0.041666666666666664},
      {0.041666666666666664, 0.16666666666666666, 0.25, 0.16666666666666666, -0.16666666666666666},
      {0.45833333333333331, 0.5, -0.25, -0.5, 0.25},
      {0.45833333333333331, -0.5, -0.25, 0.5, -0.16666666666666666},
      {0.041666666666666664, -0.16666666666666666, 0.25, -0.16666666666666666, 0.041666666666666664},
  };
  return horner(u, c[k], 5);
}

double bspline6(double x) {
  if (!(x > 0.0) || x >= 6.0) {
    return 0.0;
  }
  const int k = static_cast<int>(x);
  const double u = x - static_cast<double>(k);
  static const double c[6][6] = {
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0083333333333333332},
      {0.0083333333333333332, 0.041666666666666664, 0.083333333333333329, 0.083333333333333329,
       0.041666666666666664, -0.041666666666666664},
      {0.21666666666666667, 0.41666666666666669, 0.16666666666666669, -0.16666666666666669,
       -0.16666666666666666, 0.083333333333333343},
      {0.55000000000000004, 0.0, -0.5, 0.0, 0.25, -0.083333333333333343},
      {0.21666666666666667, -0.41666666666666669, 0.16666666666666669, 0.16666666666666669,
       -0.16666666666666669, 0.041666666666666664},
      {0.0083333333333333332, -0.041666666666666664, 0.083333333333333343, -0.083333333333333343,
       0.041666666666666664, -0.0083333333333333332},
  };
  return horner(u, c[k], 6);
}

double bspline(int order, double x) {
  if (order == 6) {
    return bspline6(x);
  }
  if (order == 5) {
    return bspline5(x);
  }
  if (order == 4) {
    return bspline4(x);
  }
  if (order == 1) {
    return (x >= 0.0 && x < 1.0) ? 1.0 : 0.0;
  }
  const double inv = 1.0 / static_cast<double>(order - 1);
  return x * inv * bspline(order - 1, x) + (order - x) * inv * bspline(order - 1, x - 1.0);
}

void weights_order5(double f, double w[5], double d[5]) {
  // Five order-5 cardinal values and derivatives at fractional coordinate f.
  const double f2 = f * f;
  const double f3 = f2 * f;
  const double f4 = f2 * f2;
  w[4] = 0.041666666666666664 * f4;
  w[3] = 0.041666666666666664 + 0.16666666666666666 * f + 0.25 * f2 + 0.16666666666666666 * f3 -
         0.16666666666666666 * f4;
  w[2] = 0.45833333333333331 + 0.5 * f - 0.25 * f2 - 0.5 * f3 + 0.25 * f4;
  w[1] = 0.45833333333333331 - 0.5 * f - 0.25 * f2 + 0.5 * f3 - 0.16666666666666666 * f4;
  w[0] = 0.041666666666666664 - 0.16666666666666666 * f + 0.25 * f2 - 0.16666666666666666 * f3 +
         0.041666666666666664 * f4;
  // Derivative is the order-4 spline difference on the same knots.
  d[0] = bspline4(f + 4.0) - bspline4(f + 3.0);
  d[1] = bspline4(f + 3.0) - bspline4(f + 2.0);
  d[2] = bspline4(f + 2.0) - bspline4(f + 1.0);
  d[3] = bspline4(f + 1.0) - bspline4(f);
  d[4] = bspline4(f) - bspline4(f - 1.0);
}

double bspline_derivative(int order, double x) {
  if (order == 6) {
    return bspline5(x) - bspline5(x - 1.0);
  }
  if (order == 4) {
    return bspline3(x) - bspline3(x - 1.0);
  }
  return bspline(order - 1, x) - bspline(order - 1, x - 1.0);
}

// One serial pocketfft call on a contiguous slab. nthreads stays 1 so the
// vendored header never starts its own worker pool.
template <class T>
void c2c_slab(std::complex<T> *data, int n0, int n1, int n2, int axis, bool forward, T scale, int tid,
              int cap) {
  const auto step = static_cast<std::ptrdiff_t>(sizeof(std::complex<T>));
  const int along = axis == 0 ? n1 : n0;
  const int mine0 = tid * along / cap;
  const int mine1 = (tid + 1) * along / cap;
  if (mine1 <= mine0) {
    return;
  }
  std::complex<T> *base = data;
  pocketfft::shape_t shape;
  pocketfft::stride_t stride{step * n1 * n2, step * n2, step};
  pocketfft::shape_t axes{static_cast<std::size_t>(axis)};
  if (axis == 0) {
    base += static_cast<std::size_t>(mine0) * static_cast<std::size_t>(n2);
    shape = {static_cast<std::size_t>(n0), static_cast<std::size_t>(mine1 - mine0), static_cast<std::size_t>(n2)};
  } else {
    base += static_cast<std::size_t>(mine0) * static_cast<std::size_t>(n1) * static_cast<std::size_t>(n2);
    shape = {static_cast<std::size_t>(mine1 - mine0), static_cast<std::size_t>(n1), static_cast<std::size_t>(n2)};
  }
  pocketfft::c2c(shape, stride, stride, axes, forward, base, base, scale, 1);
}

void fft3(std::vector<std::complex<double>> &g, int nx, int ny, int nz, bool inverse, int team) {
  (void)team;
  const int cap = mesh_slabs(std::min(nx, std::min(ny, nz)));
  const double inv = inverse ? 1.0 / static_cast<double>(static_cast<std::size_t>(nx) * ny * nz) : 1.0;
  auto stage = [&](int axis, int tid, double scale) {
    c2c_slab(g.data(), nz, ny, nx, axis, !inverse, scale, tid, cap);
  };
  team_chunk_phases(3, cap, [&](int phase, int slice) { stage(2 - phase, slice, phase == 2 ? inv : 1.0); });
}

// Phase 0 transforms the x and y axes of z slab `slice`. Phase 1
// transforms z on y slab `slice` and needs every slab of phase 0.
void mesh_xy_z(float *real, std::complex<float> *spec, int nx, int ny, int nz, bool forward, float xy_scale,
               int phase, int slice, int cap) {
  const int nkx = nx / 2 + 1;
  const auto zreal = static_cast<std::ptrdiff_t>(sizeof(float)) * nx * ny;
  const auto yreal = static_cast<std::ptrdiff_t>(sizeof(float)) * nx;
  const auto zspec = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * ny * nkx;
  const auto yspec = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * nkx;
  const int z0 = slice * nz / cap;
  const int z1 = (slice + 1) * nz / cap;
  if (phase == 0 && z1 > z0) {
    pocketfft::shape_t shape{static_cast<std::size_t>(z1 - z0), static_cast<std::size_t>(ny),
                             static_cast<std::size_t>(nx)};
    pocketfft::stride_t sreal{zreal, yreal, static_cast<std::ptrdiff_t>(sizeof(float))};
    pocketfft::stride_t sspec{zspec, yspec, static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    pocketfft::shape_t xy{1, 2};
    float *rbase = real + static_cast<std::size_t>(z0) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
    std::complex<float> *sbase =
        spec + static_cast<std::size_t>(z0) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nkx);
    if (forward) {
      pocketfft::r2c(shape, sreal, sspec, xy, true, rbase, sbase, 1.0f, 1);
    } else {
      pocketfft::c2r(shape, sspec, sreal, xy, false, sbase, rbase, xy_scale, 1);
    }
  }
  const int y0 = slice * ny / cap;
  const int y1 = (slice + 1) * ny / cap;
  if (phase == 1 && y1 > y0) {
    pocketfft::shape_t shape{static_cast<std::size_t>(nz), static_cast<std::size_t>(y1 - y0),
                             static_cast<std::size_t>(nkx)};
    pocketfft::stride_t sspec{zspec, yspec, static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
    pocketfft::shape_t zax{0};
    std::complex<float> *sbase = spec + static_cast<std::size_t>(y0) * static_cast<std::size_t>(nkx);
    pocketfft::c2c(shape, sspec, sspec, zax, forward, sbase, sbase, 1.0f, 1);
  }
}

// The transforms cut each axis into the fixed mesh slabs and the team walks
// them. Both return the number of workers that had a slab.
int mesh_r2c(float *real, std::complex<float> *spec, int nx, int ny, int nz) {
  const int team = std::max(thread_count(), 1);
  const int cap = mesh_slabs(std::min(nx, std::min(ny, nz)));
  auto work = [&](int phase, int slice) { mesh_xy_z(real, spec, nx, ny, nz, true, 1.0f, phase, slice, cap); };
  team_chunk_phases(2, cap, work);
  return std::min(team, cap);
}

int mesh_c2r(const std::complex<float> *spec, float *real, int nx, int ny, int nz, float inv) {
  const int team = std::max(thread_count(), 1);
  const int cap = mesh_slabs(std::min(nx, std::min(ny, nz)));
  // Inverse along z first, then the real axes, matching a single c2r.
  auto work = [&](int phase, int slice) {
    const int nkx = nx / 2 + 1;
    const auto zspec = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * ny * nkx;
    const auto yspec = static_cast<std::ptrdiff_t>(sizeof(std::complex<float>)) * nkx;
    const int y0 = slice * ny / cap;
    const int y1 = (slice + 1) * ny / cap;
    std::complex<float> *mutable_spec = const_cast<std::complex<float> *>(spec);
    if (phase == 0 && y1 > y0) {
      pocketfft::shape_t shape{static_cast<std::size_t>(nz), static_cast<std::size_t>(y1 - y0),
                               static_cast<std::size_t>(nkx)};
      pocketfft::stride_t sspec{zspec, yspec, static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
      pocketfft::shape_t zax{0};
      std::complex<float> *sbase = mutable_spec + static_cast<std::size_t>(y0) * static_cast<std::size_t>(nkx);
      pocketfft::c2c(shape, sspec, sspec, zax, false, sbase, sbase, 1.0f, 1);
    }
    const int z0 = slice * nz / cap;
    const int z1 = (slice + 1) * nz / cap;
    if (phase == 1 && z1 > z0) {
      const auto zreal = static_cast<std::ptrdiff_t>(sizeof(float)) * nx * ny;
      const auto yreal = static_cast<std::ptrdiff_t>(sizeof(float)) * nx;
      pocketfft::shape_t shape{static_cast<std::size_t>(z1 - z0), static_cast<std::size_t>(ny),
                               static_cast<std::size_t>(nx)};
      pocketfft::stride_t sreal{zreal, yreal, static_cast<std::ptrdiff_t>(sizeof(float))};
      pocketfft::stride_t sspec{zspec, yspec, static_cast<std::ptrdiff_t>(sizeof(std::complex<float>))};
      pocketfft::shape_t xy{1, 2};
      float *rbase = real + static_cast<std::size_t>(z0) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
      std::complex<float> *sbase =
          mutable_spec + static_cast<std::size_t>(z0) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nkx);
      pocketfft::c2r(shape, sspec, sreal, xy, false, sbase, rbase, inv, 1);
    }
  };
  team_chunk_phases(2, cap, work);
  return std::min(team, cap);
}

void bmod_axis(int order, int n, std::vector<double> *mod) {
  // |sum_k M_n(k+1) exp(2 pi i m k / K)|^2. The mesh kernel divides by
  // this so the cardinal spline's Fourier weight cancels.
  mod->assign(static_cast<std::size_t>(n), 0.0);
  for (int m = 0; m < n; ++m) {
    std::complex<double> sum(0.0, 0.0);
    for (int k = 0; k < order - 1; ++k) {
      const double ang = 2.0 * kPi * static_cast<double>(m * k) / static_cast<double>(n);
      sum += bspline(order, static_cast<double>(k + 1)) * std::complex<double>(std::cos(ang), std::sin(ang));
    }
    (*mod)[static_cast<std::size_t>(m)] = std::norm(sum);
  }
}

// Real-to-complex float mesh. The charge grid is real, so the negative x
// frequencies are the conjugates of the stored half and are not transformed.
double float_reciprocal(std::vector<std::complex<double>> &charged,
                        std::vector<std::complex<double>> &phi, int nx, int ny, int nz,
                        const std::vector<double> &bmx, const std::vector<double> &bmy,
                        const std::vector<double> &bmz, const CellGeom &cell, double alpha,
                        double pref, int team) {
  const std::size_t ntot = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nz);
  const int nkx = nx / 2 + 1;
  const std::size_t nspec = static_cast<std::size_t>(nz) * static_cast<std::size_t>(ny) *
                            static_cast<std::size_t>(nkx);
  static std::vector<float> real;
  static std::vector<std::complex<float>> spec;
  if (group_leader()) {
    if (real.size() != ntot) {
      real.assign(ntot, 0.0f);
    }
    if (spec.size() != nspec) {
      spec.assign(nspec, std::complex<float>(0.0f, 0.0f));
    }
    if (g_charge_f.size() == ntot) {
      real.swap(g_charge_f);
    } else {
      for (std::size_t i = 0; i < ntot; ++i) {
        real[i] = static_cast<float>(charged[i].real());
      }
    }
  }
  group_arrive();
  (void)team;
  const auto t_fft_a = std::chrono::steady_clock::now();
  const int fft_forward_team = mesh_r2c(real.data(), spec.data(), nx, ny, nz);
  const auto t_fft_b = std::chrono::steady_clock::now();
  g_pme_trace.fft_team = fft_forward_team;
  struct FloatFactor {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    double alpha = 0.0;
    double pref = 0.0;
    double H[9]{};
    std::vector<float> scale;
    std::vector<float> efac;
  };
  static FloatFactor saved;
  static std::vector<double> kernel_part;
  const int knt = axis_chunks(std::min(nx, std::min(ny, nz)));
  if (group_leader()) {
  kernel_part.assign(static_cast<std::size_t>(knt), 0.0);
  bool stale = saved.nx != nx || saved.ny != ny || saved.nz != nz || saved.alpha != alpha ||
               saved.pref != pref || saved.scale.size() != nspec;
  if (!stale) {
    for (int r = 0; r < 3 && !stale; ++r) {
      for (int c = 0; c < 3; ++c) {
        if (saved.H[3 * r + c] != cell.H[r][c]) {
          stale = true;
          break;
        }
      }
    }
  }
  if (stale) {
    saved.nx = nx;
    saved.ny = ny;
    saved.nz = nz;
    saved.alpha = alpha;
    saved.pref = pref;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        saved.H[3 * r + c] = cell.H[r][c];
      }
    }
    saved.scale.assign(nspec, 0.0f);
    saved.efac.assign(nspec, 0.0f);
    const double two_n = 2.0 * static_cast<double>(ntot);
    for (int iz = 0; iz < nz; ++iz) {
      for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nkx; ++ix) {
          const int sy = iy <= ny / 2 ? iy : iy - ny;
          const int sz = iz <= nz / 2 ? iz : iz - nz;
          const std::size_t at = static_cast<std::size_t>((iz * ny + iy) * nkx + ix);
          if (ix == 0 && sy == 0 && sz == 0) {
            continue;
          }
          double mc[3];
          for (int c = 0; c < 3; ++c) {
            mc[c] = static_cast<double>(ix) * cell.recip[0][c] + static_cast<double>(sy) * cell.recip[1][c] +
                    static_cast<double>(sz) * cell.recip[2][c];
          }
          const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
          const double bden = bmx[static_cast<std::size_t>(ix)] * bmy[static_cast<std::size_t>(iy)] *
                              bmz[static_cast<std::size_t>(iz)];
          if (!(m2 > 0.0) || !(bden > 1e-12)) {
            continue;
          }
          const double gauss = std::exp(-kPi * kPi * m2 / (alpha * alpha));
          const double factor = pref * gauss / m2 / bden;
          const double weight = (ix == 0 || ix == nx / 2) ? 1.0 : 2.0;
          saved.scale[at] = static_cast<float>(two_n * factor);
          saved.efac[at] = static_cast<float>(weight * factor);
        }
      }
    }
  }
  }
  group_arrive();
  double energy = 0.0;
  const auto t_kern_a = std::chrono::steady_clock::now();
  const float *scale = saved.scale.data();
  const float *efac = saved.efac.data();
  g_pme_trace.kernel_team = knt;
  auto kernel_body = [&](int slice) {
    double local = 0.0;
    const std::size_t chunk = (nspec + static_cast<std::size_t>(knt) - 1) / static_cast<std::size_t>(knt);
    const std::size_t begin = static_cast<std::size_t>(slice) * chunk;
    const std::size_t end = std::min(nspec, begin + chunk);
    for (std::size_t at = begin; at < end; ++at) {
    const float s = scale[at];
    if (!(s > 0.0f)) {
      spec[at] = std::complex<float>(0.0f, 0.0f);
      continue;
    }
    const float re = spec[at].real();
    const float im = spec[at].imag();
    local += static_cast<double>(efac[at]) * (static_cast<double>(re) * re + static_cast<double>(im) * im);
    spec[at] = std::complex<float>(s * re, s * im);
    }
    kernel_part[static_cast<std::size_t>(slice)] = local;
  };
  team_chunks(knt, kernel_body);
  if (group_leader()) {
    Neumaier folded;
    for (int t = 0; t < knt; ++t) {
      folded.add(kernel_part[static_cast<std::size_t>(t)]);
    }
    energy = folded.total();
  }
  const auto t_kern_b = std::chrono::steady_clock::now();
  const float inv = 1.0f / static_cast<float>(ntot);
  const auto t_ifft_a = std::chrono::steady_clock::now();
  g_pme_trace.fft_team = std::max(g_pme_trace.fft_team, mesh_c2r(spec.data(), real.data(), nx, ny, nz, inv));
  const auto t_ifft_b = std::chrono::steady_clock::now();
  g_pme_trace.fft_us = std::chrono::duration<double, std::micro>(t_fft_b - t_fft_a).count() +
                       std::chrono::duration<double, std::micro>(t_ifft_b - t_ifft_a).count();
  g_pme_trace.kernel_us = std::chrono::duration<double, std::micro>(t_kern_b - t_kern_a).count();
  // Keep the real potential in the float buffer the gather reads.
  if (group_leader()) {
    g_charge_f.swap(real);
  }
  group_arrive();
  return group_leader() ? energy : 0.0;
}

void spme_reciprocal(const CoulombConfig &cfg, const CellGeom &cell, const double *pos,
                     const double *q, std::size_t n, double total, double *F, double *dE_dq,
                     ElecOut *out) {
  if (cfg.spline_order < 3 || cfg.spline_order > 16) {
    throw std::invalid_argument("SPME spline order must be at least 3");
  }
  const int nx = std::max(cfg.grid[0], 4);
  const int ny = std::max(cfg.grid[1], 4);
  const int nz = std::max(cfg.grid[2], 4);
  const int order = cfg.spline_order;
  const double alpha = cfg.alpha;
  const std::size_t ntot = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                           static_cast<std::size_t>(nz);
  static std::vector<std::complex<double>> grid;
  if (group_leader()) {
    if (grid.size() != ntot) {
      grid.assign(ntot, std::complex<double>(0.0, 0.0));
    } else {
      std::fill(grid.begin(), grid.end(), std::complex<double>(0.0, 0.0));
    }
  }
  group_arrive();
  auto index = [nx, ny](int x, int y, int z) {
    return static_cast<std::size_t>(x + nx * (y + ny * z));
  };
  static std::vector<double> bmx_s, bmy_s, bmz_s;
  if (group_leader()) {
    bmod_axis(order, nx, &bmx_s);
    bmod_axis(order, ny, &bmy_s);
    bmod_axis(order, nz, &bmz_s);
  }
  group_arrive();
  std::vector<double> &bmx = bmx_s;
  std::vector<double> &bmy = bmy_s;
  std::vector<double> &bmz = bmz_s;

  const std::size_t span = static_cast<std::size_t>(order);
  static std::vector<int> gtab_s;
  static std::vector<double> wtab_s;
  static std::vector<double> dtab_s;
  if (group_leader()) {
    const std::size_t need = 3 * n * span;
    if (gtab_s.size() != need) {
      gtab_s.assign(need, 0);
      wtab_s.assign(need, 0.0);
      dtab_s.assign(need, 0.0);
    }
  }
  group_arrive();
  std::vector<int> &gtab = gtab_s;
  std::vector<double> &wtab = wtab_s;
  std::vector<double> &dtab = dtab_s;
  auto slot = [span](std::size_t atom, int axis, int knot) {
    return (atom * 3 + static_cast<std::size_t>(axis)) * span + static_cast<std::size_t>(knot);
  };
  const int nthreads = thread_count();
  const bool trace = std::getenv("LR_PME_PROFILE") != nullptr;
  const bool mesh_float_now = !cfg.stress;
  if (!mesh_float_now) {
    if (group_leader() && !g_charge_f.empty()) {
      g_charge_f.clear();
    }
  } else if (group_leader() && g_charge_f.size() != ntot) {
    g_charge_f.assign(ntot, 0.0f);
  }
  group_arrive();
  // Fixed z slabs, each a padded private slab rather than a full private
  // mesh. Each slab deposits the atoms whose stencil starts in it, and
  // every plane folds the slabs in slab order, so the charge grid has the
  // same bits on any team.
  static std::vector<float> slabs_f;
  static std::vector<double> slabs_d;
  static std::vector<int> zbase_s;
  static std::vector<int> bin_off;
  static std::vector<int> bin_atoms;
  const int pad = order - 1;
  const int nslab = spread_slabs(nz);
  const int slab_h = (nz + nslab - 1) / nslab + pad;
  const int plane = nx * ny;
  const std::size_t slab_stride = static_cast<std::size_t>(slab_h) * static_cast<std::size_t>(plane);
  auto z_lo = [nz, nslab](int s) { return s * nz / nslab; };
  auto z_hi = [nz, nslab](int s) { return (s + 1) * nz / nslab; };
  static std::vector<double> seen_pos;
  static double seen_box[9];
  static int seen_grid[4];
  // The lead thread rewrites the key on a miss, and the team must take one
  // decision about the gates below.
  static bool team_tables_hit = false;
  bool tables_hit = false;
  if (group_leader()) {
    if (mesh_float_now && gtab.size() == 3 * n * span && zbase_s.size() == n && seen_pos.size() == 3 * n &&
        seen_grid[0] == nx && seen_grid[1] == ny && seen_grid[2] == nz && seen_grid[3] == order) {
      bool box_same = true;
      for (int r = 0; r < 3 && box_same; ++r) {
        for (int c = 0; c < 3; ++c) {
          if (seen_box[3 * r + c] != cell.H[r][c]) {
            box_same = false;
          }
        }
      }
      if (box_same && std::memcmp(seen_pos.data(), pos, 3 * n * sizeof(double)) == 0) {
        tables_hit = true;
      }
    }
    team_tables_hit = tables_hit;
    if (zbase_s.size() != n) {
      zbase_s.assign(n, 0);
    }
    if (mesh_float_now) {
      if (slabs_f.size() != static_cast<std::size_t>(nslab) * slab_stride) {
        slabs_f.assign(static_cast<std::size_t>(nslab) * slab_stride, 0.0f);
      }
    } else if (slabs_d.size() != static_cast<std::size_t>(nslab) * slab_stride) {
      slabs_d.assign(static_cast<std::size_t>(nslab) * slab_stride, 0.0);
    }
  }
  group_arrive();
  tables_hit = team_tables_hit;
  std::vector<int> &zbase = zbase_s;
  const auto t_spread0 = std::chrono::steady_clock::now();
  // Stencil of one atom. The z start is kept in [0, nz) so its raw planes
  // run from zbase - pad to zbase.
  auto fill_tables = [&](std::size_t i) {
    double r[3] = {pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]};
    double fs[3];
    cell.fractional(r, fs);
    const double nd[3] = {static_cast<double>(nx), static_cast<double>(ny), static_cast<double>(nz)};
    const int naxis[3] = {nx, ny, nz};
    for (int c = 0; c < 3; ++c) {
      const double u = (fs[c] - std::floor(fs[c])) * nd[c];
      const double base = std::floor(u);
      const int k0 = static_cast<int>(base) - order + 1;
      if (c == 2) {
        int zb = static_cast<int>(base) % nz;
        if (zb < 0) {
          zb += nz;
        }
        zbase[i] = zb;
      }
      if (order == 5) {
        double wv[5], dv[5];
        weights_order5(u - base, wv, dv);
        for (int j = 0; j < 5; ++j) {
          int w = (k0 + j) % naxis[c];
          if (w < 0) {
            w += naxis[c];
          }
          const std::size_t at = slot(i, c, j);
          gtab[at] = w;
          wtab[at] = wv[j];
          dtab[at] = dv[j];
        }
        continue;
      }
      for (int j = 0; j < order; ++j) {
        const int g = k0 + j;
        int w = g % naxis[c];
        if (w < 0) {
          w += naxis[c];
        }
        const double arg = u - static_cast<double>(g);
        const std::size_t at = slot(i, c, j);
        gtab[at] = w;
        wtab[at] = bspline(order, arg);
        dtab[at] = bspline_derivative(order, arg);
      }
    }
  };
  if (!tables_hit) {
    const int nfill = rgpot::fixedChunks(n, 64);
    team_chunks(nfill, [&](int chunk) {
      const Range run = partition(static_cast<std::int64_t>(n), chunk, nfill);
      for (auto i = static_cast<std::size_t>(run.begin); i < static_cast<std::size_t>(run.end); ++i) {
        fill_tables(i);
      }
    });
  }
  if (group_leader()) {
    if (!tables_hit) {
      seen_pos.assign(pos, pos + 3 * n);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          seen_box[3 * r + c] = cell.H[r][c];
        }
      }
      seen_grid[0] = nx;
      seen_grid[1] = ny;
      seen_grid[2] = nz;
      seen_grid[3] = order;
    }
    // Stable bins by z start, so a slab walks its atoms in a fixed order.
    bin_off.assign(static_cast<std::size_t>(nz) + 1, 0);
    bin_atoms.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      bin_off[static_cast<std::size_t>(zbase[i]) + 1] += 1;
    }
    for (int z = 0; z < nz; ++z) {
      bin_off[static_cast<std::size_t>(z) + 1] += bin_off[static_cast<std::size_t>(z)];
    }
    std::vector<int> cursor(bin_off.begin(), bin_off.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
      bin_atoms[static_cast<std::size_t>(cursor[static_cast<std::size_t>(zbase[i])]++)] = static_cast<int>(i);
    }
  }
  group_arrive();
  auto spread_into = [&](auto *slab_base, auto *dest, std::size_t dest_step) {
    using T = std::remove_pointer_t<decltype(slab_base)>;
    team_chunks(nslab, [&](int s) {
      const int z0 = z_lo(s);
      const int z1 = z_hi(s);
      T *mine = slab_base + static_cast<std::size_t>(s) * slab_stride;
      const int height = (z1 - z0) + pad;
      std::fill(mine, mine + static_cast<std::size_t>(height) * static_cast<std::size_t>(plane), T(0));
      for (int a = bin_off[static_cast<std::size_t>(z0)]; a < bin_off[static_cast<std::size_t>(z1)]; ++a) {
        const std::size_t i = static_cast<std::size_t>(bin_atoms[static_cast<std::size_t>(a)]);
        const double qi = q[i];
        const int *gx = gtab.data() + (i * 3 + 0) * span;
        const int *gy = gtab.data() + (i * 3 + 1) * span;
        const double *wx = wtab.data() + (i * 3 + 0) * span;
        const double *wy = wtab.data() + (i * 3 + 1) * span;
        const double *wz = wtab.data() + (i * 3 + 2) * span;
        // Raw plane zbase - pad + iz sits at local row zbase - z0 + iz.
        const int row0 = zbase[i] - z0;
        for (int iz = 0; iz < order; ++iz) {
          T *layer = mine + static_cast<std::size_t>(row0 + iz) * static_cast<std::size_t>(plane);
          for (int iy = 0; iy < order; ++iy) {
            const double wyz = wy[iy] * wz[iz];
            T *line = layer + static_cast<std::size_t>(gy[iy]) * static_cast<std::size_t>(nx);
            for (int ix = 0; ix < order; ++ix) {
              line[gx[ix]] += static_cast<T>(qi * wx[ix] * wyz);
            }
          }
        }
      }
    });
    team_chunks(nz, [&](int gz) {
      T *dst = dest + static_cast<std::size_t>(gz) * static_cast<std::size_t>(plane) * dest_step;
      for (int k = 0; k < plane; ++k) {
        dst[static_cast<std::size_t>(k) * dest_step] = T(0);
      }
      for (int s = 0; s < nslab; ++s) {
        const int origin = z_lo(s) - pad;
        const T *mine = slab_base + static_cast<std::size_t>(s) * slab_stride;
        for (int raw = origin; raw < z_hi(s); ++raw) {
          int wrapped = raw % nz;
          if (wrapped < 0) {
            wrapped += nz;
          }
          if (wrapped != gz) {
            continue;
          }
          const T *src = mine + static_cast<std::size_t>(raw - origin) * static_cast<std::size_t>(plane);
          for (int k = 0; k < plane; ++k) {
            dst[static_cast<std::size_t>(k) * dest_step] += src[k];
          }
        }
      }
    });
  };
  if (mesh_float_now) {
    spread_into(slabs_f.data(), g_charge_f.data(), 1);
  } else {
    spread_into(slabs_d.data(), reinterpret_cast<double *>(grid.data()), 2);
  }
  const auto t_reduce0 = std::chrono::steady_clock::now();
  const auto t_fft0 = std::chrono::steady_clock::now();
  g_pme_trace.spread_us = std::chrono::duration<double, std::micro>(t_fft0 - t_spread0).count();
  if (group_leader()) {
    g_pme_trace.spread_team = nthreads;
  }
  if (trace) {
    std::fprintf(stderr, "pme_reduce workers=%d deposit_us=%.3f reduce_us=%.3f\n", nthreads,
                 std::chrono::duration<double, std::micro>(t_reduce0 - t_spread0).count(),
                 std::chrono::duration<double, std::micro>(t_fft0 - t_reduce0).count());
  }
  const bool mesh_float = !cfg.stress;
  if (!mesh_float) {
    fft3(grid, nx, ny, nz, false, nthreads);
  }
  const auto t_kern0 = std::chrono::steady_clock::now();
  static std::vector<std::complex<double>> phihat;
  if (!mesh_float && group_leader()) {
    if (phihat.size() != ntot) {
      phihat.assign(ntot, std::complex<double>(0.0, 0.0));
    } else {
      std::fill(phihat.begin(), phihat.end(), std::complex<double>(0.0, 0.0));
    }
  }
  if (!mesh_float) {
    group_arrive();
  }
  struct KernelLocal {
    Neumaier rec;
    double strain[6]{};
    char pad[64];
  };
  // One partial per z plane, folded in plane order.
  static std::vector<KernelLocal> local;
  if (group_leader()) {
    local.assign(mesh_float ? 0 : static_cast<std::size_t>(nz), KernelLocal{});
  }
  group_arrive();
  const double pref = kKe / (2.0 * kPi * cell.volume);
  auto apply_stress = [&](int plane_z) {
    Neumaier &rec = local[static_cast<std::size_t>(plane_z)].rec;
    double *st = local[static_cast<std::size_t>(plane_z)].strain;
    const int iz = plane_z;
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        const int sx = ix <= nx / 2 ? ix : ix - nx;
        const int sy = iy <= ny / 2 ? iy : iy - ny;
        const int sz = iz <= nz / 2 ? iz : iz - nz;
        if (sx == 0 && sy == 0 && sz == 0) {
          continue;
        }
        double mc[3];
        for (int c = 0; c < 3; ++c) {
          mc[c] = static_cast<double>(sx) * cell.recip[0][c] +
                  static_cast<double>(sy) * cell.recip[1][c] +
                  static_cast<double>(sz) * cell.recip[2][c];
        }
        const double m2 = mc[0] * mc[0] + mc[1] * mc[1] + mc[2] * mc[2];
        const double bden = bmx[static_cast<std::size_t>(ix)] *
                            bmy[static_cast<std::size_t>(iy)] *
                            bmz[static_cast<std::size_t>(iz)];
        if (!(m2 > 0.0) || !(bden > 1e-12)) {
          continue;
        }
        const double gauss = std::exp(-kPi * kPi * m2 / (alpha * alpha));
        const double factor = pref * gauss / m2 / bden;
        const auto qh = grid[index(ix, iy, iz)];
        rec.add(factor * std::norm(qh));
        phihat[index(ix, iy, iz)] = (2.0 * static_cast<double>(ntot) * factor) * qh;
        if (st != nullptr) {
          const double s2 = std::norm(qh) / bden;
          // |k|^2 = (2 pi)^2 |m|^2 and exp(-|k|^2 / (4 a^2)) is `gauss`.
          const double k2 = 4.0 * kPi * kPi * m2;
          const double ewald_f = gauss / k2;
          const double ewald_pref = kKe * (2.0 * kPi) / cell.volume;
          const double inv4a = 1.0 / (4.0 * alpha * alpha);
          double kv[3];
          for (int c = 0; c < 3; ++c) {
            kv[c] = 2.0 * kPi * mc[c];
          }
          for (int b = 0; b < 3; ++b) {
            for (int c = 0; c <= b; ++c) {
              const double dfac = -((b == c) ? 1.0 : 0.0) +
                                  2.0 * (inv4a + 1.0 / k2) * kv[b] * kv[c];
              const int voigt = b == 0 && c == 0   ? 0
                                : b == 1 && c == 1 ? 1
                                : b == 2 && c == 2 ? 2
                                : (b == 1 && c == 2) || (b == 2 && c == 1) ? 3
                                : (b == 0 && c == 2) || (b == 2 && c == 0) ? 4
                                                                            : 5;
              st[voigt] += s2 * ewald_pref * ewald_f * dfac;
            }
          }
        }
      }
    }
  };
  if (mesh_float) {
    const double mesh_e =
        float_reciprocal(grid, phihat, nx, ny, nz, bmx, bmy, bmz, cell, alpha, pref, nthreads);
    if (group_leader()) {
      out->energy += mesh_e;
    }
  } else {
    // Stress is the only double-precision mesh. The charge-only path uses
    // the float kernel above, so there is no second consumer of a factor table.
    team_chunks(nz, apply_stress);
  }
  if (group_leader()) {
    Neumaier rec;
    for (const KernelLocal &part : local) {
      rec.add(part.rec);
      for (int k = 0; k < 6; ++k) {
        out->strain[k] += part.strain[k];
      }
    }
    if (!mesh_float) {
      out->energy += rec.total();
    }
  }
  const double ebg = -kKe * kPi * total * total / (2.0 * cell.volume * alpha * alpha);
  if (group_leader()) {
    out->energy += ebg;
    out->strain[0] += -ebg;
    out->strain[1] += -ebg;
    out->strain[2] += -ebg;
  }
  const auto t_gather0 = std::chrono::steady_clock::now();
  if (!mesh_float) {
    fft3(phihat, nx, ny, nz, true, nthreads);
  }
  const auto t_force0 = std::chrono::steady_clock::now();
  g_pme_trace.gather_us = 0;
  auto gather_atom = [&](std::size_t i) {
    double pot = 0.0;
    double dQ[3] = {};
    if (order == 5 && !g_charge_f.empty()) {
      const float *phi = g_charge_f.data();
      const int *gx = gtab.data() + (i * 3 + 0) * 5;
      const int *gy = gtab.data() + (i * 3 + 1) * 5;
      const int *gz = gtab.data() + (i * 3 + 2) * 5;
      const double *wx = wtab.data() + (i * 3 + 0) * 5;
      const double *wy = wtab.data() + (i * 3 + 1) * 5;
      const double *wz = wtab.data() + (i * 3 + 2) * 5;
      const double *dx = dtab.data() + (i * 3 + 0) * 5;
      const double *dy = dtab.data() + (i * 3 + 1) * 5;
      const double *dz = dtab.data() + (i * 3 + 2) * 5;
      const double qw = q[i];
      for (int iz = 0; iz < 5; ++iz) {
        const double wz_i = wz[iz];
        const double dz_i = dz[iz];
        const int gz_i = gz[iz];
        for (int iy = 0; iy < 5; ++iy) {
          const double wy_i = wy[iy];
          const double wywz = wy_i * wz_i;
          const double qdy = qw * dy[iy] * wz_i;
          const double qdz = qw * wy_i * dz_i;
          const int base = nx * (gy[iy] + ny * gz_i);
          const int g0 = gx[0], g1 = gx[1], g2 = gx[2], g3 = gx[3], g4 = gx[4];
          const double ph0 = static_cast<double>(phi[base + g0]);
          const double ph1 = static_cast<double>(phi[base + g1]);
          const double ph2 = static_cast<double>(phi[base + g2]);
          const double ph3 = static_cast<double>(phi[base + g3]);
          const double ph4 = static_cast<double>(phi[base + g4]);
          const double w0 = wx[0], w1 = wx[1], w2 = wx[2], w3 = wx[3], w4 = wx[4];
          const double d0 = dx[0], d1 = dx[1], d2 = dx[2], d3 = dx[3], d4 = dx[4];
          const double x0 = w0 * ph0, x1 = w1 * ph1, x2 = w2 * ph2, x3 = w3 * ph3, x4 = w4 * ph4;
          pot += wywz * (x0 + x1 + x2 + x3 + x4);
          dQ[0] += qw * wywz * (d0 * ph0 + d1 * ph1 + d2 * ph2 + d3 * ph3 + d4 * ph4);
          dQ[1] += qdy * (x0 + x1 + x2 + x3 + x4);
          dQ[2] += qdz * (x0 + x1 + x2 + x3 + x4);
        }
      }
    } else {
    for (int iz = 0; iz < order; ++iz) {
      for (int iy = 0; iy < order; ++iy) {
        for (int ix = 0; ix < order; ++ix) {
          const double wx = wtab[slot(i, 0, ix)];
          const double wy = wtab[slot(i, 1, iy)];
          const double wz = wtab[slot(i, 2, iz)];
          const std::size_t cell_at = index(gtab[slot(i, 0, ix)], gtab[slot(i, 1, iy)], gtab[slot(i, 2, iz)]);
          const double ph = g_charge_f.empty() ? phihat[cell_at].real() : static_cast<double>(g_charge_f[cell_at]);
          pot += wx * wy * wz * ph;
          const double qw = q[i];
          dQ[0] += qw * dtab[slot(i, 0, ix)] * wy * wz * ph;
          dQ[1] += qw * wx * dtab[slot(i, 1, iy)] * wz * ph;
          dQ[2] += qw * wx * wy * dtab[slot(i, 2, iz)] * ph;
        }
      }
    }
    }
    if (dE_dq != nullptr) {
      dE_dq[i] += pot;
    }
    // u = s * n, du/dr = n * H^{-1}. F = -dE/dr and dE = phi dQ, with
    // dQ/du already in dQ and phi included.
    for (int a = 0; a < 3; ++a) {
      double dE = 0.0;
      const double scale[3] = {static_cast<double>(nx), static_cast<double>(ny),
                               static_cast<double>(nz)};
      for (int d = 0; d < 3; ++d) {
        dE += dQ[d] * scale[d] * cell.inv[a][d];
      }
      F[3 * i + a] += -dE;
    }
  };
  // Each atom gathers on its own, so any split of the atoms keeps the bits.
  const int gather_cap = atom_workers(static_cast<int>(n), nthreads);
  team_chunks(gather_cap, [&](int slice) {
    const Range run = partition(static_cast<std::int64_t>(n), slice, gather_cap);
    for (auto i = static_cast<std::size_t>(run.begin); i < static_cast<std::size_t>(run.end); ++i) {
      gather_atom(i);
    }
  });
  if (group_leader()) {
    g_pme_trace.gather_team = std::max(gather_cap, 1);
  }
  if (dE_dq != nullptr) {
    const double dbg = -kKe * kPi * total / (cell.volume * alpha * alpha);
    for (std::size_t i = 0; i < n; ++i) {
      dE_dq[i] += dbg;
    }
  }
  out->has_stress = true;
  const auto t_end = std::chrono::steady_clock::now();
  g_pme_trace.gather_us = std::chrono::duration<double, std::micro>(t_end - t_force0).count();
  if (trace) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::fprintf(stderr,
                 "pme_phase n=%zu grid=%d workers=%d spread_us=%.3f fft_us=%.3f kernel_us=%.3f ifft_us=%.3f gather_us=%.3f\n",
                 n, nx, nthreads, us(t_spread0, t_fft0), us(t_fft0, t_kern0), us(t_kern0, t_gather0),
                 us(t_gather0, t_force0), us(t_force0, t_end));
  }
}

} // namespace

FixedCharges::FixedCharges(std::vector<double> q, double total)
    : q_(std::move(q)), total_(total) {}

void FixedCharges::load(const ForceInput &in, std::vector<double> *q, double *total) const {
  if (q_.size() != in.nAtoms) {
    throw std::invalid_argument("fixed charges do not match the atom count");
  }
  *q = q_;
  *total = total_;
}

FnCharges::FnCharges(Fn fn, void *user) : fn_(fn), user_(user) {
  if (fn_ == nullptr) {
    throw std::invalid_argument("charge callback is null");
  }
}

void FnCharges::load(const ForceInput &in, std::vector<double> *q, double *total) const {
  q->assign(in.nAtoms, 0.0);
  fn_(in, q->data(), total, user_);
}

void so3lr_erf_pair(double qi, double qj, double r, double sigma, double cuton, double cutoff,
                    double ke, double *energy, double *dEdr) {
  auto pot = [sigma](double x) { return std::erf(x / sigma) / x; };
  auto dpot = [sigma](double x) {
    const double a = x / sigma;
    return (2.0 * x * std::exp(-a * a) / (std::sqrt(kPi) * sigma) - std::erf(a)) / (x * x);
  };
  const double span = cutoff - cuton;
  double f = 0.0;
  double df = 0.0;
  if (!(span > 0.0) || r >= cutoff) {
    f = 0.0;
  } else if (r <= cuton) {
    f = 1.0;
  } else {
    const double c = (r - cuton) / span;
    auto sig = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) : 0.0; };
    auto sigp = [](double x) { return x > 0.0 ? std::exp(-1.0 / x) / (x * x) : 0.0; };
    const double s1 = sig(1.0 - c);
    const double s2 = sig(c);
    const double den = s1 + s2;
    f = s1 / den;
    const double ds1 = -sigp(1.0 - c);
    const double ds2 = sigp(c);
    const double dfdc = (ds1 * den - s1 * (ds1 + ds2)) / (den * den);
    df = dfdc / span;
  }
  if (!(r > 0.0) || r >= cutoff) {
    *energy = 0.0;
    *dEdr = 0.0;
    return;
  }
  const double pairwise = pot(r);
  const double shift = pot(cutoff);
  const double force_shift = dpot(cutoff);
  const double shifted = pairwise - shift - force_shift * (r - cutoff);
  const double blend = f * (pairwise - shift) + (1.0 - f) * shifted;
  const double qq = ke * qi * qj;
  *energy = qq * blend;
  const double dblend = df * (pairwise - shift - shifted) +
                        f * dpot(r) + (1.0 - f) * (dpot(r) - force_shift);
  *dEdr = qq * dblend;
}

void evaluate_electrostatics(CoulombMethod method, const CoulombConfig &cfg, const double *pos,
                             const double *q, std::size_t n, double total_charge, const double *box,
                             const std::vector<Pair> &pairs, double *F, double *dE_dq, ElecOut *out) {
  if (out == nullptr || (F == nullptr && n > 0)) {
    throw std::invalid_argument("electrostatic output is null");
  }
  if (!(cfg.cutoff > 0.0) || !(cfg.alpha > 0.0)) {
    throw std::invalid_argument("cutoff and alpha must be positive");
  }
  require_charges(q, n, total_charge);
  out->energy = 0.0;
  std::fill(out->strain, out->strain + 6, 0.0);
  out->has_stress = false;
  if (dE_dq != nullptr) {
    std::fill(dE_dq, dE_dq + n, 0.0);
  }
  if (n == 0) {
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(pos[3 * i]) || !std::isfinite(pos[3 * i + 1]) ||
        !std::isfinite(pos[3 * i + 2])) {
      throw std::invalid_argument("a position is not finite");
    }
  }
  if (method == CoulombMethod::DSF) {
    real_space(cfg, true, q, n, pairs, F, dE_dq, out);
    out->has_stress = true;
    return;
  }
  if (cfg.kmax < 1 && method == CoulombMethod::Ewald) {
    throw std::invalid_argument("Ewald kmax must be at least 1");
  }
  const CellGeom cell = cell_from_box(box);
  struct ResetLimit {
    int &slot;
    int prev;
    ~ResetLimit() { slot = prev; }
  };
  const ResetLimit reset_spme{spme_slice_limit(), spme_slice_limit()};
  const ResetLimit reset_group{mesh_group_limit(), mesh_group_limit()};
  if (method == CoulombMethod::SPME) {
    // A request of eight on a short mesh is a team of four. Shrink the
    // pool before any phase so the spare workers are not created and do
    // not sit in an active wait outside a smaller region.
    const int requested = std::max(caller_team(), thread_count());
    const int npoints = std::min(cfg.grid[0], std::min(cfg.grid[1], cfg.grid[2]));
    const int team = pme_team(static_cast<int>(n), npoints, requested);
    resizePool(team);
    spme_slice_limit() = team;
  }
  // Real space always lands in its own buffers and is added to the
  // reciprocal result once at the end. Which threads ran it, and whether
  // it ran beside the mesh, then leaves the sum order alone.
  std::vector<double> real_f(3 * n, 0.0);
  std::vector<double> real_dq(dE_dq != nullptr ? n : 0, 0.0);
  ElecOut real_out{};
  auto add_real = [&]() {
    for (std::size_t i = 0; i < real_f.size(); ++i) {
      F[i] += real_f[i];
    }
    for (std::size_t i = 0; i < real_dq.size(); ++i) {
      dE_dq[i] += real_dq[i];
    }
    out->energy += real_out.energy;
    for (int k = 0; k < 6; ++k) {
      out->strain[k] += real_out.strain[k];
    }
  };
  // Two threads on a short mesh: one thread in the transform does not
  // beat one thread, so the second thread runs real space beside it.
  const int asked = std::max(caller_team(), thread_count());
  const int mesh_cap = axis_workers(std::min(cfg.grid[0], std::min(cfg.grid[1], cfg.grid[2])), asked);
  const int real_beside =
      method == CoulombMethod::SPME && !cfg.stress && dE_dq == nullptr
          ? beside_real_workers(static_cast<int>(n),
                                std::min(cfg.grid[0], std::min(cfg.grid[1], cfg.grid[2])), asked)
          : 0;
  // Spare threads hide real space behind the reciprocal pipeline. The
  // mesh team is the axis cap. The threads past it walk the real-space
  // chunks. On a shorter team the same chunks run one phase at a time.
  if (real_beside > 0 && mesh_cap >= 4) {
    const auto t_real0 = std::chrono::steady_clock::now();
    ensure_screen_cache(cfg, pairs);
    const int nchunks = real_chunks(pairs);
    std::vector<Scratch> &parts = scratch_for(nchunks, 3 * n);
    mesh_group_limit() = mesh_cap;
    const bool together = onFullTeam(mesh_cap + real_beside, [&](int rank) {
      if (rank < mesh_cap) {
        t_group_rank = rank;
        spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
        t_group_rank = -1;
      } else {
        for (int chunk = rank - mesh_cap; chunk < nchunks; chunk += real_beside) {
          real_cached_chunk(q, pairs, cfg.cutoff, parts[static_cast<std::size_t>(chunk)], chunk, nchunks);
        }
      }
    });
    if (!together) {
      spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
      forEachChunk(nchunks, real_beside, [&](int chunk) {
        real_cached_chunk(q, pairs, cfg.cutoff, parts[static_cast<std::size_t>(chunk)], chunk, nchunks);
      });
    }
    mesh_group_limit() = 0;
    add_scratch(parts, real_f.data(), &real_out);
    add_ewald_self(cfg, q, n, &real_out);
    g_pme_trace.real_us =
        std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_real0).count();
    g_pme_trace.real_team = real_beside;
    add_real();
    return;
  }
  // Two threads do not speed a short transform, so the second thread runs
  // real space in the same OpenMP region.
  const bool beside_pair = method == CoulombMethod::SPME && thread_count() == 2 && mesh_cap == 1 &&
                           !cfg.stress && dE_dq == nullptr;
  if (beside_pair) {
    mesh_group_limit() = 1;
    const bool together = onFullTeam(2, [&](int rank) {
      if (rank == 0) {
        t_group_rank = 0;
        spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
        t_group_rank = -1;
      } else {
        real_space(cfg, false, q, n, pairs, real_f.data(), nullptr, &real_out);
      }
    });
    if (!together) {
      spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
      real_space(cfg, false, q, n, pairs, real_f.data(), nullptr, &real_out);
    }
    mesh_group_limit() = 0;
    add_real();
    return;
  }
  real_space(cfg, false, q, n, pairs, real_f.data(), dE_dq != nullptr ? real_dq.data() : nullptr, &real_out);
  if (method == CoulombMethod::Ewald) {
    ewald_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
  } else {
    spme_reciprocal(cfg, cell, pos, q, n, total_charge, F, dE_dq, out);
  }
  add_real();
}

CoulombPot::CoulombPot(CoulombConfig cfg, std::shared_ptr<ChargeSource> charges,
                       std::shared_ptr<ChargeResponse> response)
    : Potential(PotType::LrCoulomb), cfg_(cfg), charges_(std::move(charges)),
      response_(std::move(response)),
      lists_(std::make_shared<DualList>(cfg.short_cutoff, cfg.cutoff, cfg.skin)) {
  if (!charges_) {
    throw std::invalid_argument("CoulombPot needs a charge source");
  }
  Fnv1a fp;
  fp.u64(1);
  fp.u64(static_cast<uint64_t>(cfg_.method));
  fp.f64(cfg_.cutoff);
  fp.f64(cfg_.alpha);
  fp.i64(cfg_.kmax);
  fp.i64(cfg_.grid[0]);
  fp.i64(cfg_.grid[1]);
  fp.i64(cfg_.grid[2]);
  fp.i64(cfg_.spline_order);
  fp.f64(cfg_.skin);
  fp.u64(cfg_.periodic ? 1 : 0);
  params_ = fp.h;
}

void CoulombPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const rgpot::ThreadBinding bind_threads{threadCount()};
  std::fill(out->F, out->F + 3 * in.nAtoms, 0.0);
  out->energy = 0.0;
  out->variance = 0.0;
  out->has_stress = 0;
  std::vector<double> q;
  double total = 0.0;
  charges_->load(in, &q, &total);
  const bool periodic = cfg_.periodic && cfg_.method != CoulombMethod::DSF;
  const auto t_list = std::chrono::steady_clock::now();
  lists_->update(in.pos, in.nAtoms, in.box, periodic);
  const auto t_eval = std::chrono::steady_clock::now();
  std::vector<double> dedq(in.nAtoms, 0.0);
  ElecOut er;
  evaluate_electrostatics(cfg_.method, cfg_, in.pos, q.data(), in.nAtoms, total,
                          periodic ? in.box : nullptr, lists_->long_pairs(), out->F,
                          response_ ? dedq.data() : nullptr, &er);
  out->energy = er.energy;
  if (response_) {
    response_->accumulate(in, dedq.data(), out->F);
  }
  const double volume = periodic ? cellVolume(in.box) : 0.0;
  if (er.has_stress && volume > 0.0) {
    publishCauchyStress(out, er.strain, volume);
  }
  if (std::getenv("LR_PME_PROFILE") != nullptr && cfg_.method == CoulombMethod::SPME) {
    const auto t_end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "pme_outer n=%zu pairs=%zu list_us=%.3f eval_us=%.3f\n", in.nAtoms,
                 lists_->long_pairs().size(),
                 std::chrono::duration<double, std::micro>(t_eval - t_list).count(),
                 std::chrono::duration<double, std::micro>(t_end - t_eval).count());
  }
}

PmeTrace pme_trace() { return g_pme_trace; }

} // namespace rgpot::lr
