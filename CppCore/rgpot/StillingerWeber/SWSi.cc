// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file SWSi.cc
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Stillinger-Weber silicon kernel, 1985 parameters.
 */

#include "rgpot/StillingerWeber/SWSi.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/StillingerWeber/SWInternal.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace {

using Bond = swi::Bond;

struct PairTerm {
  bool live;
  double f2;
  double dedr;
  double gexp;
  double dgdr;
};

PairTerm pairTerm(double r) {
  const double u = r / StillingerWeberSi::kSigma;
  const double gap = u - StillingerWeberSi::kCut;
  // The factor exp(1/(u-a)) underflows to zero before u reaches a.
  if (!(gap < 0.0) || gap > -1.0 / 700.0) {
    return {};
  }
  const double s = 1.0 / gap;
  const double inv_u = 1.0 / u;
  const double u2 = inv_u * inv_u;
  const double u_m4 = u2 * u2;
  const double g = StillingerWeberSi::kB * u_m4 - 1.0;
  const double es = std::exp(s);
  const double f2 = StillingerWeberSi::kA * g * es;
  const double dgdu =
      -StillingerWeberSi::kP * StillingerWeberSi::kB * u_m4 * inv_u;
  const double dfdu = StillingerWeberSi::kA * es * (dgdu - g * s * s);
  const double dedr =
      StillingerWeberSi::kEpsilon * dfdu / StillingerWeberSi::kSigma;
  const double gexp = std::exp(StillingerWeberSi::kGamma * s);
  const double dgdr =
      gexp * (-StillingerWeberSi::kGamma * s * s) / StillingerWeberSi::kSigma;
  return {true, f2, dedr, gexp, dgdr};
}

struct ForceHit {
  int atom;
  double x;
  double y;
  double z;
};

// Hits already sit in scatter order, one source atom after another.
// Grouping them by the atom they update keeps that order inside the row,
// and each thread sums the atoms it owns.
[[gnu::noinline]] void gatherByAtom(const int *force_at, const int *force_used,
                                    const ForceHit *hits, double *acc, int n, int nt) {
  struct Scratch {
    std::vector<int> local_n;
    std::vector<int> touch;
    std::vector<int> cursor;
    std::vector<ForceHit> csr;
  };
  thread_local Scratch scratch;
  auto &local_n = scratch.local_n;
  auto &touch = scratch.touch;
  auto &cursor = scratch.cursor;
  auto &csr = scratch.csr;
  if (static_cast<int>(local_n.size()) < nt * n) {
    local_n.resize(static_cast<std::size_t>(nt) * static_cast<std::size_t>(n));
  }
  std::fill(local_n.begin(),
            local_n.begin() + static_cast<std::size_t>(nt) * static_cast<std::size_t>(n), 0);
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    int *mine = local_n.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(n);
    const auto range = nlist::partition(n, t, nt);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const ForceHit *hit = hits + force_at[i];
      const int nf = force_used[i];
      for (int k = 0; k < nf; ++k) {
        ++mine[hit[k].atom];
      }
    }
  }
  touch.resize(static_cast<std::size_t>(n) + 1);
  touch[0] = 0;
  for (int atom = 0; atom < n; ++atom) {
    int count = 0;
    for (int t = 0; t < nt; ++t) {
      count += local_n[static_cast<std::size_t>(t) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(atom)];
    }
    touch[static_cast<std::size_t>(atom) + 1] = touch[static_cast<std::size_t>(atom)] + count;
  }
  const int ntouch = touch[static_cast<std::size_t>(n)];
  if (static_cast<int>(csr.size()) < ntouch) {
    csr.resize(static_cast<std::size_t>(ntouch));
  }
  if (static_cast<int>(cursor.size()) < nt * n) {
    cursor.resize(static_cast<std::size_t>(nt) * static_cast<std::size_t>(n));
  }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    int *place = cursor.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(n);
    for (int atom = 0; atom < n; ++atom) {
      int off = touch[static_cast<std::size_t>(atom)];
      for (int u = 0; u < t; ++u) {
        off += local_n[static_cast<std::size_t>(u) * static_cast<std::size_t>(n) +
                       static_cast<std::size_t>(atom)];
      }
      place[atom] = off;
    }
    const auto range = nlist::partition(n, t, nt);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const ForceHit *hit = hits + force_at[i];
      const int nf = force_used[i];
      for (int k = 0; k < nf; ++k) {
        csr[static_cast<std::size_t>(place[hit[k].atom])] = hit[k];
        ++place[hit[k].atom];
      }
    }
  }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    const auto range = nlist::partition(n, t, nt);
    for (int atom = static_cast<int>(range.begin); atom < static_cast<int>(range.end); ++atom) {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      const int begin = touch[static_cast<std::size_t>(atom)];
      const int end = touch[static_cast<std::size_t>(atom) + 1];
      for (int p = begin; p < end; ++p) {
        x += csr[static_cast<std::size_t>(p)].x;
        y += csr[static_cast<std::size_t>(p)].y;
        z += csr[static_cast<std::size_t>(p)].z;
      }
      acc[3 * atom] += x;
      acc[3 * atom + 1] += y;
      acc[3 * atom + 2] += z;
    }
  }
}

} // namespace

StillingerWeberSi::StillingerWeberSi() : Potential(PotType::SWSi1985) {
  Fnv1a fp;
  fp.u64(1);
  fp.f64(kEpsilon);
  fp.f64(kSigma);
  fp.f64(kA);
  fp.f64(kB);
  fp.f64(kP);
  fp.f64(kCut);
  fp.f64(kLambda);
  fp.f64(kGamma);
  m_paramsKey = fp.h;
}

void StillingerWeberSi::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("StillingerWeberSi: null ForceOut");
  }
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  if (in.nAtoms > 0 &&
      (in.pos == nullptr || in.box == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("StillingerWeberSi: null force buffer");
  }
  evaluate(in.pos, in.box, in.nAtoms, &out->energy, out->F, 1.0, nullptr);
}

void StillingerWeberSi::energyAndGradient(
    const double *positions, const int *atomic_numbers, const double *box,
    std::size_t n_atoms, double *energy, double *gradient,
    nlist::NeighborList *neighbors) const {
  (void)atomic_numbers;
  if (energy == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradient == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null energy buffer");
  }
  // Gradient is dE/dx, the negation of the force written by forceImpl.
  evaluate(positions, box, n_atoms, energy, gradient, -1.0, neighbors);
}

void StillingerWeberSi::energyAndGradientBatch(
    std::size_t n_config, const double *positions, const int *atomic_numbers,
    const double *box, std::size_t n_atoms, double *energies,
    double *gradients) const {
  if (n_config == 0) {
    return;
  }
  if (energies == nullptr ||
      (n_atoms > 0 &&
       (positions == nullptr || box == nullptr || gradients == nullptr))) {
    throw std::invalid_argument("StillingerWeberSi: null batch buffer");
  }
  const std::size_t stride = n_atoms * 3;
  for (std::size_t c = 0; c < n_config; ++c) {
    energyAndGradient(positions + c * stride, atomic_numbers, box, n_atoms,
                      energies + c, gradients + c * stride);
  }
}

void StillingerWeberSi::evaluate(const double *positions, const double *box,
                                 std::size_t n_atoms, double *energy,
                                 double *acc, double scale,
                                 nlist::NeighborList *neighbors) const {
  *energy = 0.0;
  const std::size_t n3 = n_atoms * 3;
  for (std::size_t i = 0; i < n3; ++i) {
    acc[i] = 0.0;
  }
  if (n_atoms < 2) {
    return;
  }

  const double cut = kCut * kSigma;
  const double cut2 = cut * cut;
  const int n = static_cast<int>(n_atoms);

  const double skin = (neighbors != nullptr && neighbors->skin > 0.0)
                          ? neighbors->skin
                          : (mode_ == nlist::Mode::Stream ? 0.0 : cut / 5.0);
  const auto before = neighbours_.builds();
  nlist::updateCells(neighbours_, positions, n_atoms, box, cut, skin, nlist::Image::Round,
                     mode_, {true, true, true});
  if (neighbors != nullptr && neighbors->skin > 0.0 &&
      (neighbours_.builds() != before || !neighbors->built)) {
    neighbors->ref.assign(positions, positions + 3 * n_atoms);
    std::memcpy(neighbors->box, box, 9 * sizeof(double));
    neighbors->n = n_atoms;
    neighbors->cutoff = cut;
    neighbors->skin_used = neighbors->skin;
    neighbors->built = true;
  }

  // One atom of bonds. A stream reads the bins directly, so the scratch
  // is that atom and not a copy of every bond. A stored call reads the
  // index row, which is already paid for.
  auto load = [&](int i, std::vector<Bond> &nb) {
    nb.clear();
    auto take = [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
      if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz)) {
        throw std::invalid_argument("StillingerWeberSi: non-finite position");
      }
      if (!(r2 > 0.0) || r2 >= cut2) {
        return;
      }
      const double r = std::sqrt(r2);
      const PairTerm t = pairTerm(r);
      if (!t.live) {
        return;
      }
      nb.push_back(Bond{static_cast<int>(j), -dx, -dy, -dz, r, t.gexp, t.dgdr});
    };
    if (neighbours_.streaming()) {
      neighbours_.forAtom(static_cast<std::int32_t>(i), take);
    } else {
      nlist::visitDirected(neighbours_, i, static_cast<std::int64_t>(i) + 1, take);
    }
    std::sort(nb.begin(), nb.end(), [](const Bond &a, const Bond &b) { return a.j < b.j; });
  };
  auto pairForces = [&](int i, const std::vector<Bond> &nb, double *force,
                        double &local_e) {
    for (const Bond &bj : nb) {
      if (bj.j <= i) {
        continue;
      }
      const PairTerm t = pairTerm(bj.r);
      if (!t.live) {
        continue;
      }
      local_e += kEpsilon * t.f2;
      const double inv_r = 1.0 / bj.r;
      const double sx = scale * t.dedr * bj.x * inv_r;
      const double sy = scale * t.dedr * bj.y * inv_r;
      const double sz = scale * t.dedr * bj.z * inv_r;
      force[3 * i] += sx;
      force[3 * i + 1] += sy;
      force[3 * i + 2] += sz;
      force[3 * bj.j] -= sx;
      force[3 * bj.j + 1] -= sy;
      force[3 * bj.j + 2] -= sz;
    }
  };

  const int nt = nlist::threadsRequested();
  if (nt == 1) {
  std::vector<Bond> nb;
  for (int i = 0; i < n; ++i) {
    load(i, nb);
    pairForces(i, nb, acc, *energy);
    const int m = static_cast<int>(nb.size());
    for (int a = 0; a < m; ++a) {
      const Bond &bj = nb[static_cast<std::size_t>(a)];
      for (int b = a + 1; b < m; ++b) {
        const Bond &bk = nb[static_cast<std::size_t>(b)];
        const double inv_r1 = 1.0 / bj.r;
        const double inv_r2 = 1.0 / bk.r;
        const double inv = inv_r1 * inv_r2;
        double cth = (bj.x * bk.x + bj.y * bk.y + bj.z * bk.z) * inv;
        if (cth > 1.0) {
          cth = 1.0;
        } else if (cth < -1.0) {
          cth = -1.0;
        }
        const double ang = cth + (1.0 / 3.0);
        const double ang2 = ang * ang;
        *energy += kEpsilon * kLambda * bj.gexp * bk.gexp * ang2;
        const double pref = kEpsilon * kLambda;
        const double d_edcth = pref * bj.gexp * bk.gexp * 2.0 * ang;
        const double d_edr1 = pref * bj.dgdr * bk.gexp * ang2;
        const double d_edr2 = pref * bj.gexp * bk.dgdr * ang2;
        const double g1x =
            d_edr1 * bj.x * inv_r1 +
            d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
        const double g1y =
            d_edr1 * bj.y * inv_r1 +
            d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
        const double g1z =
            d_edr1 * bj.z * inv_r1 +
            d_edcth * (bk.z * inv - cth * bj.z * inv_r1 * inv_r1);
        const double g2x =
            d_edr2 * bk.x * inv_r2 +
            d_edcth * (bj.x * inv - cth * bk.x * inv_r2 * inv_r2);
        const double g2y =
            d_edr2 * bk.y * inv_r2 +
            d_edcth * (bj.y * inv - cth * bk.y * inv_r2 * inv_r2);
        const double g2z =
            d_edr2 * bk.z * inv_r2 +
            d_edcth * (bj.z * inv - cth * bk.z * inv_r2 * inv_r2);
        // g is dE/d(neighbor). Force on that atom is -g.
        acc[3 * bj.j] -= scale * g1x;
        acc[3 * bj.j + 1] -= scale * g1y;
        acc[3 * bj.j + 2] -= scale * g1z;
        acc[3 * bk.j] -= scale * g2x;
        acc[3 * bk.j + 1] -= scale * g2y;
        acc[3 * bk.j + 2] -= scale * g2z;
        acc[3 * i] += scale * (g1x + g2x);
        acc[3 * i + 1] += scale * (g1y + g2y);
        acc[3 * i + 2] += scale * (g1z + g2z);
      }
    }
  }

  } else {
    // Each thread writes the pair and triplet updates into slots whose
    // index is their place in the one-thread scatter. The gather adds
    // those slots in that order.
    struct Slots {
      std::vector<Bond> bonds;
      std::vector<int> bond_at;
      std::vector<int> bond_n;
      std::vector<int> energy_at;
      std::vector<int> energy_n;
      std::vector<int> energy_used;
      std::vector<int> force_at;
      std::vector<int> force_n;
      std::vector<int> force_used;
      std::vector<double> energy;
      std::vector<ForceHit> force;
    };
    thread_local Slots slots;
    auto &bonds = slots.bonds;
    auto &bond_at = slots.bond_at;
    auto &bond_n = slots.bond_n;
    auto &energy_at = slots.energy_at;
    auto &energy_n = slots.energy_n;
    auto &energy_used = slots.energy_used;
    auto &force_at = slots.force_at;
    auto &force_n = slots.force_n;
    auto &force_used = slots.force_used;
    auto &energy_slot = slots.energy;
    auto &force = slots.force;
    const auto &row = neighbours_.offsets();
    const bool stored = !neighbours_.streaming() &&
                        row.size() == static_cast<std::size_t>(n) + 1;
    // Below 1024 atoms the copy and the serial slot scan stay as they are.
    // At 1024 and above the wide kernel writes each force into its atom's row.
    if (stored && n >= 1024) {
      swi::runWide(neighbours_, n, nt, cut2, scale, energy, acc);
    } else {
    bond_at.resize(static_cast<std::size_t>(n) + 1);
    bond_n.assign(static_cast<std::size_t>(n), 0);
    bond_at[0] = 0;
    if (stored) {
      for (int i = 0; i < n; ++i) {
        const int deg = row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
        bond_at[static_cast<std::size_t>(i) + 1] =
            bond_at[static_cast<std::size_t>(i)] + deg;
      }
      if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
        bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
      }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
      {
#if defined(_OPENMP)
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        const auto range = nlist::partition(n, t, nt);
        std::vector<Bond> nb;
        for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
          load(i, nb);
          const int room = bond_at[static_cast<std::size_t>(i) + 1] -
                           bond_at[static_cast<std::size_t>(i)];
          const int m = std::min(static_cast<int>(nb.size()), room);
          std::copy(nb.begin(), nb.begin() + m,
                    bonds.begin() + bond_at[static_cast<std::size_t>(i)]);
          bond_n[static_cast<std::size_t>(i)] = m;
        }
      }
    } else {
      bond_at[static_cast<std::size_t>(n)] = 0;
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
      {
#if defined(_OPENMP)
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        const auto range = nlist::partition(n, t, nt);
        std::vector<Bond> nb;
        for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
          load(i, nb);
          bond_n[static_cast<std::size_t>(i)] = static_cast<int>(nb.size());
        }
      }
      for (int i = 0; i < n; ++i) {
        bond_at[static_cast<std::size_t>(i) + 1] =
            bond_at[static_cast<std::size_t>(i)] + bond_n[static_cast<std::size_t>(i)];
      }
      if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
        bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
      }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
      {
#if defined(_OPENMP)
        const int t = omp_get_thread_num();
#else
        const int t = 0;
#endif
        const auto range = nlist::partition(n, t, nt);
        std::vector<Bond> nb;
        for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
          load(i, nb);
          const int m = bond_n[static_cast<std::size_t>(i)];
          std::copy(nb.begin(), nb.begin() + m,
                    bonds.begin() + bond_at[static_cast<std::size_t>(i)]);
        }
      }
    }
    energy_at.resize(static_cast<std::size_t>(n) + 1);
    force_at.resize(static_cast<std::size_t>(n) + 1);
    energy_n.assign(static_cast<std::size_t>(n), 0);
    energy_used.assign(static_cast<std::size_t>(n), 0);
    force_n.assign(static_cast<std::size_t>(n), 0);
    force_used.assign(static_cast<std::size_t>(n), 0);
    energy_at[0] = 0;
    force_at[0] = 0;
    for (int i = 0; i < n; ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      int pairs = 0;
      for (int a = 0; a < m; ++a) {
        if (nb[a].j > i) {
          ++pairs;
        }
      }
      const int trips = m * (m - 1) / 2;
      energy_n[static_cast<std::size_t>(i)] = pairs + trips;
      force_n[static_cast<std::size_t>(i)] = 2 * pairs + 3 * trips;
      energy_at[static_cast<std::size_t>(i) + 1] =
          energy_at[static_cast<std::size_t>(i)] + pairs + trips;
      force_at[static_cast<std::size_t>(i) + 1] =
          force_at[static_cast<std::size_t>(i)] + 2 * pairs + 3 * trips;
    }
    const int nenergy = energy_at[static_cast<std::size_t>(n)];
    const int nforce = force_at[static_cast<std::size_t>(n)];
    if (static_cast<int>(energy_slot.size()) < nenergy) {
      energy_slot.resize(static_cast<std::size_t>(nenergy));
    }
    if (static_cast<int>(force.size()) < nforce) {
      force.resize(static_cast<std::size_t>(nforce));
    }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int t = omp_get_thread_num();
#else
      const int t = 0;
#endif
      const auto range = nlist::partition(n, t, nt);
      for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
        const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
        const int m = bond_n[static_cast<std::size_t>(i)];
        double *e_steps = energy_slot.data() + energy_at[static_cast<std::size_t>(i)];
        ForceHit *f_steps = force.data() + force_at[static_cast<std::size_t>(i)];
        int ne = 0;
        int nf = 0;
        for (int a = 0; a < m; ++a) {
          const Bond &bj = nb[a];
          if (bj.j <= i) {
            continue;
          }
          const PairTerm term = pairTerm(bj.r);
          if (!term.live) {
            continue;
          }
          e_steps[ne] = kEpsilon * term.f2;
          ++ne;
          const double inv_r = 1.0 / bj.r;
          const double sx = scale * term.dedr * bj.x * inv_r;
          const double sy = scale * term.dedr * bj.y * inv_r;
          const double sz = scale * term.dedr * bj.z * inv_r;
          f_steps[nf] = ForceHit{i, sx, sy, sz};
          ++nf;
          f_steps[nf] = ForceHit{bj.j, -sx, -sy, -sz};
          ++nf;
        }
        for (int a = 0; a < m; ++a) {
          const Bond &bj = nb[a];
          for (int b = a + 1; b < m; ++b) {
            const Bond &bk = nb[b];
            const double inv_r1 = 1.0 / bj.r;
            const double inv_r2 = 1.0 / bk.r;
            const double inv = inv_r1 * inv_r2;
            double cth = (bj.x * bk.x + bj.y * bk.y + bj.z * bk.z) * inv;
            if (cth > 1.0) {
              cth = 1.0;
            } else if (cth < -1.0) {
              cth = -1.0;
            }
            const double ang = cth + (1.0 / 3.0);
            const double ang2 = ang * ang;
            e_steps[ne] = kEpsilon * kLambda * bj.gexp * bk.gexp * ang2;
            ++ne;
            const double pref = kEpsilon * kLambda;
            const double d_edcth = pref * bj.gexp * bk.gexp * 2.0 * ang;
            const double d_edr1 = pref * bj.dgdr * bk.gexp * ang2;
            const double d_edr2 = pref * bj.gexp * bk.dgdr * ang2;
            const double g1x =
                d_edr1 * bj.x * inv_r1 + d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
            const double g1y =
                d_edr1 * bj.y * inv_r1 + d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
            const double g1z =
                d_edr1 * bj.z * inv_r1 + d_edcth * (bk.z * inv - cth * bj.z * inv_r1 * inv_r1);
            const double g2x =
                d_edr2 * bk.x * inv_r2 + d_edcth * (bj.x * inv - cth * bk.x * inv_r2 * inv_r2);
            const double g2y =
                d_edr2 * bk.y * inv_r2 + d_edcth * (bj.y * inv - cth * bk.y * inv_r2 * inv_r2);
            const double g2z =
                d_edr2 * bk.z * inv_r2 + d_edcth * (bj.z * inv - cth * bk.z * inv_r2 * inv_r2);
            f_steps[nf] = ForceHit{bj.j, -scale * g1x, -scale * g1y, -scale * g1z};
            ++nf;
            f_steps[nf] = ForceHit{bk.j, -scale * g2x, -scale * g2y, -scale * g2z};
            ++nf;
            f_steps[nf] = ForceHit{i, scale * (g1x + g2x), scale * (g1y + g2y),
                                   scale * (g1z + g2z)};
            ++nf;
          }
        }
        energy_used[static_cast<std::size_t>(i)] = ne;
        force_used[static_cast<std::size_t>(i)] = nf;
      }
    }
    for (int i = 0; i < n; ++i) {
      const double *step = energy_slot.data() + energy_at[static_cast<std::size_t>(i)];
      const int ne = energy_used[static_cast<std::size_t>(i)];
      for (int k = 0; k < ne; ++k) {
        *energy += step[k];
      }
    }
    // Below 1024 atoms the serial replay is the fast path. At 1024 and
    // above each thread sums the atoms it owns, in scatter order.
    if (n < 1024) {
      for (int i = 0; i < n; ++i) {
        const ForceHit *hit = force.data() + force_at[static_cast<std::size_t>(i)];
        const int nf = force_used[static_cast<std::size_t>(i)];
        for (int k = 0; k < nf; ++k) {
          acc[3 * hit[k].atom] += hit[k].x;
          acc[3 * hit[k].atom + 1] += hit[k].y;
          acc[3 * hit[k].atom + 2] += hit[k].z;
        }
      }
    } else {
      gatherByAtom(force_at.data(), force_used.data(), force.data(), acc, n, nt);
    }
    }
  }
}

} // namespace rgpot
