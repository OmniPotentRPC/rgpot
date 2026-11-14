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

int storeBonds(nlist::Neighbours &cells, int i, double cut2, Bond *dest, int room) {
  int m = 0;
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
    if (m >= room) {
      return;
    }
    dest[m] = Bond{static_cast<int>(j), -dx, -dy, -dz, r, t.gexp, t.dgdr};
    ++m;
  };
  if (cells.streaming()) {
    cells.forAtom(static_cast<std::int32_t>(i), take);
  } else {
    nlist::visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, take);
  }
  std::sort(dest, dest + m, [](const Bond &a, const Bond &b) { return a.j < b.j; });
  return m;
}

struct Hit {
  int atom;
  double x;
  double y;
  double z;
};

struct WideScratch {
  std::vector<Bond> bonds;
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

// Pair and triplet targets, in the order the fill emits them. Threads own
// contiguous atom ranges, so a prefix across threads is the serial order.
void countTargets(const std::vector<Bond> &bonds, const std::vector<int> &bond_at,
                  const std::vector<int> &bond_n, std::vector<int> &energy_at,
                  std::vector<int> &energy_n, std::vector<int> &local_n, std::vector<int> &touch,
                  int n, int nt) {
  energy_at.resize(static_cast<std::size_t>(n) + 1);
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
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      int pairs = 0;
      for (int a = 0; a < m; ++a) {
        if (nb[a].j > i) {
          ++pairs;
          ++mine[i];
          ++mine[nb[a].j];
        }
      }
      for (int a = 0; a < m; ++a) {
        for (int b = a + 1; b < m; ++b) {
          ++mine[nb[a].j];
          ++mine[nb[b].j];
          ++mine[i];
        }
      }
      const int trips = m * (m - 1) / 2;
      energy_n[static_cast<std::size_t>(i)] = pairs + trips;
    }
  }
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
  bond_n.assign(static_cast<std::size_t>(n), 0);
  bond_at[0] = 0;
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
      bond_n[static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
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
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const int room =
          bond_at[static_cast<std::size_t>(i) + 1] - bond_at[static_cast<std::size_t>(i)];
      Bond *dest = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      bond_n[static_cast<std::size_t>(i)] = storeBonds(cells, i, cut2, dest, room);
    }
  }
  countTargets(bonds, bond_at, bond_n, energy_at, energy_n, local_n, touch, n, nt);
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
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
      int ne = 0;
      for (int a = 0; a < m; ++a) {
        const Bond &bj = nb[a];
        if (bj.j <= i) {
          continue;
        }
        const PairTerm term = pairTerm(bj.r);
        if (!term.live) {
          continue;
        }
        steps[ne] = StillingerWeberSi::kEpsilon * term.f2;
        ++ne;
        const double inv_r = 1.0 / bj.r;
        const double sx = scale * term.dedr * bj.x * inv_r;
        const double sy = scale * term.dedr * bj.y * inv_r;
        const double sz = scale * term.dedr * bj.z * inv_r;
        csr[static_cast<std::size_t>(place[i]++)] = Hit{i, sx, sy, sz};
        csr[static_cast<std::size_t>(place[bj.j]++)] = Hit{bj.j, -sx, -sy, -sz};
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
              Hit{bj.j, -scale * g1x, -scale * g1y, -scale * g1z};
          csr[static_cast<std::size_t>(place[bk.j]++)] =
              Hit{bk.j, -scale * g2x, -scale * g2y, -scale * g2z};
          csr[static_cast<std::size_t>(place[i]++)] =
              Hit{i, scale * (g1x + g2x), scale * (g1y + g2y), scale * (g1z + g2z)};
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
