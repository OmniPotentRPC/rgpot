#include "rgpot/StillingerWeber/SWInternal.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace swi {
namespace {

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
  const double dgdu = -StillingerWeberSi::kP * StillingerWeberSi::kB * u_m4 * inv_u;
  const double dfdu = StillingerWeberSi::kA * es * (dgdu - g * s * s);
  const double dedr = StillingerWeberSi::kEpsilon * dfdu / StillingerWeberSi::kSigma;
  const double gexp = std::exp(StillingerWeberSi::kGamma * s);
  const double dgdr = gexp * (-StillingerWeberSi::kGamma * s * s) / StillingerWeberSi::kSigma;
  return {true, f2, dedr, gexp, dgdr};
}

int storeBonds(nlist::Neighbours &cells, int i, double cut2, Bond *dest, double *f2,
               double *dedr, int room) {
  struct Item {
    Bond bond;
    double f2;
    double dedr;
  };
  // One buffer per worker. The sort has to carry the pair term with the bond.
  thread_local std::vector<Item> items;
  items.clear();
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
    if (static_cast<int>(items.size()) >= room) {
      return;
    }
    items.push_back(
        Item{Bond{static_cast<int>(j), -dx, -dy, -dz, r, t.gexp, t.dgdr}, t.f2, t.dedr});
  };
  if (cells.streaming()) {
    cells.forAtom(static_cast<std::int32_t>(i), take);
  } else {
    nlist::visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, take);
  }
  std::sort(items.begin(), items.end(),
            [](const Item &a, const Item &b) { return a.bond.j < b.bond.j; });
  const int m = static_cast<int>(items.size());
  for (int k = 0; k < m; ++k) {
    dest[k] = items[static_cast<std::size_t>(k)].bond;
    f2[k] = items[static_cast<std::size_t>(k)].f2;
    dedr[k] = items[static_cast<std::size_t>(k)].dedr;
  }
  return m;
}

// The row is the atom, so the gather does not keep a copy of the index.
struct Hit {
  double x;
  double y;
  double z;
};

struct WideScratch {
  std::vector<Bond> bonds;
  std::vector<double> pair_f2;
  std::vector<double> pair_dedr;
  std::vector<int> bond_at;
  std::vector<int> bond_n;
  std::vector<int> energy_at;
  std::vector<int> energy_n;
  std::vector<int> energy_used;
  std::vector<double> energy;
  std::vector<int> local_n;
  std::vector<int> touch;
  std::vector<int> cursor;
  std::vector<Hit> csr;
};

// Threads already counted while the bonds were loaded. A prefix across
// those counts is the serial order, because each thread owns a lower
// contiguous range than the next.
void prefixTargets(const std::vector<int> &energy_n, const std::vector<int> &local_n,
                   std::vector<int> &energy_at, std::vector<int> &touch, int n, int nt) {
  energy_at.resize(static_cast<std::size_t>(n) + 1);
  energy_at[0] = 0;
  touch.resize(static_cast<std::size_t>(n) + 1);
  touch[0] = 0;
  for (int i = 0; i < n; ++i) {
    energy_at[static_cast<std::size_t>(i) + 1] =
        energy_at[static_cast<std::size_t>(i)] + energy_n[static_cast<std::size_t>(i)];
    int hits = 0;
    for (int t = 0; t < nt; ++t) {
      hits += local_n[static_cast<std::size_t>(t) * static_cast<std::size_t>(n) +
                      static_cast<std::size_t>(i)];
    }
    touch[static_cast<std::size_t>(i) + 1] = touch[static_cast<std::size_t>(i)] + hits;
  }
}

} // namespace

void runWide(nlist::Neighbours &cells, int n, int nt, double cut2, double scale, double *energy,
             double *acc) {
  thread_local WideScratch scratch;
  auto &bonds = scratch.bonds;
  auto &pair_f2 = scratch.pair_f2;
  auto &pair_dedr = scratch.pair_dedr;
  auto &bond_at = scratch.bond_at;
  auto &bond_n = scratch.bond_n;
  auto &energy_at = scratch.energy_at;
  auto &energy_n = scratch.energy_n;
  auto &energy_used = scratch.energy_used;
  auto &eterm = scratch.energy;
  auto &local_n = scratch.local_n;
  auto &touch = scratch.touch;
  auto &cursor = scratch.cursor;
  auto &csr = scratch.csr;
  const auto &row = cells.offsets();
  bond_at.resize(static_cast<std::size_t>(n) + 1);
  bond_n.resize(static_cast<std::size_t>(n));
  bond_at[0] = 0;
  for (int i = 0; i < n; ++i) {
    const int deg = row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
    bond_n[static_cast<std::size_t>(i)] = deg;
    bond_at[static_cast<std::size_t>(i) + 1] = bond_at[static_cast<std::size_t>(i)] + deg;
  }
  if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
    bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
    pair_f2.resize(bonds.size());
    pair_dedr.resize(bonds.size());
  }
  energy_n.assign(static_cast<std::size_t>(n), 0);
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
      const int room =
          bond_at[static_cast<std::size_t>(i) + 1] - bond_at[static_cast<std::size_t>(i)];
      Bond *dest = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m =
          storeBonds(cells, i, cut2, dest, pair_f2.data() + bond_at[static_cast<std::size_t>(i)],
                     pair_dedr.data() + bond_at[static_cast<std::size_t>(i)], room);
      bond_n[static_cast<std::size_t>(i)] = m;
      int pairs = 0;
      for (int a = 0; a < m; ++a) {
        const int j = dest[a].j;
        // Every other neighbour makes one triplet with this one.
        mine[j] += m - 1;
        if (j > i) {
          ++pairs;
          ++mine[j];
        }
      }
      const int trips = m * (m - 1) / 2;
      mine[i] += pairs + trips;
      energy_n[static_cast<std::size_t>(i)] = pairs + trips;
    }
  }
  prefixTargets(energy_n, local_n, energy_at, touch, n, nt);
  const int nenergy = energy_at[static_cast<std::size_t>(n)];
  const int ntouch = touch[static_cast<std::size_t>(n)];
  if (static_cast<int>(eterm.size()) < nenergy) {
    eterm.resize(static_cast<std::size_t>(nenergy));
  }
  energy_used.assign(static_cast<std::size_t>(n), 0);
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
    // Each bond's reciprocal is shared by every angle at that bond.
    // A longer neighbour list keeps the same values in a temporary buffer.
    constexpr int kReciprocalCap = 256;
    double inv_stack[kReciprocalCap];
    std::vector<double> inv_heap;
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
      int ne = 0;
      double *inv_r = inv_stack;
      if (m > kReciprocalCap) {
        inv_heap.resize(static_cast<std::size_t>(m));
        inv_r = inv_heap.data();
      }
      for (int a = 0; a < m; ++a) {
        inv_r[a] = 1.0 / nb[a].r;
      }
      for (int a = 0; a < m; ++a) {
        const Bond &bj = nb[a];
        if (bj.j <= i) {
          continue;
        }
        const double f2 = pair_f2[bond_at[static_cast<std::size_t>(i)] + a];
        const double dedr = pair_dedr[bond_at[static_cast<std::size_t>(i)] + a];
        steps[ne] = StillingerWeberSi::kEpsilon * f2;
        ++ne;
        const double inv = inv_r[static_cast<std::size_t>(a)];
        const double sx = scale * dedr * bj.x * inv;
        const double sy = scale * dedr * bj.y * inv;
        const double sz = scale * dedr * bj.z * inv;
        csr[static_cast<std::size_t>(place[i]++)] = Hit{sx, sy, sz};
        csr[static_cast<std::size_t>(place[bj.j]++)] = Hit{-sx, -sy, -sz};
      }
      for (int a = 0; a < m; ++a) {
        const Bond &bj = nb[a];
        const double inv_r1 = inv_r[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < m; ++b) {
          const Bond &bk = nb[b];
          const double inv_r2 = inv_r[static_cast<std::size_t>(b)];
          const double inv = inv_r1 * inv_r2;
          double cth = (bj.x * bk.x + bj.y * bk.y + bj.z * bk.z) * inv;
          if (cth > 1.0) {
            cth = 1.0;
          } else if (cth < -1.0) {
            cth = -1.0;
          }
          const double ang = cth + (1.0 / 3.0);
          const double ang2 = ang * ang;
          steps[ne] = StillingerWeberSi::kEpsilon * StillingerWeberSi::kLambda * bj.gexp *
                      bk.gexp * ang2;
          ++ne;
          const double pref = StillingerWeberSi::kEpsilon * StillingerWeberSi::kLambda;
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
          csr[static_cast<std::size_t>(place[bj.j]++)] =
              Hit{-scale * g1x, -scale * g1y, -scale * g1z};
          csr[static_cast<std::size_t>(place[bk.j]++)] =
              Hit{-scale * g2x, -scale * g2y, -scale * g2z};
          csr[static_cast<std::size_t>(place[i]++)] =
              Hit{scale * (g1x + g2x), scale * (g1y + g2y), scale * (g1z + g2z)};
        }
      }
      energy_used[static_cast<std::size_t>(i)] = ne;
    }
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
    const int ne = energy_used[static_cast<std::size_t>(i)];
    for (int k = 0; k < ne; ++k) {
      total += steps[k];
    }
  }
  *energy = total;
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

} // namespace swi
} // namespace rgpot
