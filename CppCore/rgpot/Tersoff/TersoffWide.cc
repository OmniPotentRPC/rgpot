#include "rgpot/Tersoff/TersoffSiC.hpp"

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct Bond {
  int j;
  double x, y, z, r;
  double fc, dfc, f_r, f_a, df_r, df_a;
};

struct Angle {
  double cth;
  double g;
  double dg;
};

struct Hit {
  double x;
  double y;
  double z;
};

struct StrainHit {
  int kind;
  double p0, p1, p2, p3, p4, p5;
};

void angular(const TersoffSiC::Species &sp, double cth, double &g, double &dg) {
  const double gap = sp.h - cth;
  const double den = sp.d2 + gap * gap;
  g = sp.gamma * (1.0 + sp.c2 / sp.d2 - sp.c2 / den);
  dg = sp.gamma * (-2.0 * sp.c2 * gap) / (den * den);
}

void bondOrder(const TersoffSiC::Species &sp, double zeta, double &b, double &db) {
  if (zeta == 0.0) {
    b = 1.0;
    db = 0.0;
    return;
  }
  const double base = 1.0 + sp.beta_n * std::pow(zeta, sp.n);
  b = std::pow(base, -0.5 / sp.n);
  db = -0.5 * std::pow(base, -0.5 / sp.n - 1.0) * sp.beta_n * std::pow(zeta, sp.n - 1.0);
}

void radial(const TersoffSiC::Pair &p, double r, Bond &b) {
  if (r <= p.rin) {
    b.fc = 1.0;
    b.dfc = 0.0;
  } else {
    const double width = p.rout - p.rin;
    const double phase = kPi * (r - p.rin) / width;
    b.fc = 0.5 + 0.5 * std::cos(phase);
    b.dfc = -0.5 * kPi / width * std::sin(phase);
  }
  const double e1 = std::exp(-p.lambda1 * r);
  const double e2 = std::exp(-p.lambda2 * r);
  b.f_r = p.big_a * e1;
  b.df_r = -p.lambda1 * b.f_r;
  b.f_a = -p.big_b * e2;
  b.df_a = -p.lambda2 * b.f_a;
}

struct RadialMemo {
  double r = -1.0;
  const TersoffSiC::Pair *pair = nullptr;
  double fc = 0;
  double dfc = 0;
  double f_r = 0;
  double f_a = 0;
  double df_r = 0;
  double df_a = 0;
  void apply(const TersoffSiC::Pair &p, double radius, Bond &b) {
    if (pair == &p && r == radius) {
      b.fc = fc;
      b.dfc = dfc;
      b.f_r = f_r;
      b.f_a = f_a;
      b.df_r = df_r;
      b.df_a = df_a;
      return;
    }
    radial(p, radius, b);
    pair = &p;
    r = radius;
    fc = b.fc;
    dfc = b.dfc;
    f_r = b.f_r;
    f_a = b.f_a;
    df_r = b.df_r;
    df_a = b.df_a;
  }
};

struct OrderMemo {
  const TersoffSiC::Species *sp = nullptr;
  double zeta = -1.0;
  double b = 0;
  double db = 0;
  void apply(const TersoffSiC::Species &species, double z, double &b_out, double &db_out) {
    if (sp == &species && zeta == z) {
      b_out = b;
      db_out = db;
      return;
    }
    bondOrder(species, z, b_out, db_out);
    sp = &species;
    zeta = z;
    b = b_out;
    db = db_out;
  }
};

double cosine(const Bond &a, const Bond &b);

// The cosine is symmetric, so each angle is stored for both bond orders.
// Zeta is still summed in neighbour order.
void fillAngles(const Bond *nb, int m, const TersoffSiC::Species &sp, Angle *ang) {
  for (int a = 0; a < m; ++a) {
    for (int k = a + 1; k < m; ++k) {
      Angle slot;
      slot.cth = cosine(nb[a], nb[k]);
      angular(sp, slot.cth, slot.g, slot.dg);
      ang[static_cast<std::size_t>(a * m + k)] = slot;
      ang[static_cast<std::size_t>(k * m + a)] = slot;
    }
  }
}

double cosine(const Bond &a, const Bond &b) {
  const double inv = 1.0 / (a.r * b.r);
  double c = (a.x * b.x + a.y * b.y + a.z * b.z) * inv;
  if (c > 1.0) {
    c = 1.0;
  } else if (c < -1.0) {
    c = -1.0;
  }
  return c;
}

int storeBonds(nlist::Neighbours &cells, const TersoffSiC::Pair pairs[2][2], const int *kind,
               int i, double cut2, Bond *dest, int room, RadialMemo &radial_memo) {
  int m = 0;
  const int ki = kind[i];
  auto take = [&](std::int32_t, std::int32_t j, double dx, double dy, double dz, double r2) {
    if (!(r2 > 0.0) || r2 >= cut2) {
      return;
    }
    const double r = std::sqrt(r2);
    const TersoffSiC::Pair &pij = pairs[ki][kind[j]];
    if (r >= pij.rout) {
      return;
    }
    if (m >= room) {
      return;
    }
    Bond forth{static_cast<int>(j), -dx, -dy, -dz, r, 0, 0, 0, 0, 0, 0};
    radial_memo.apply(pij, r, forth);
    dest[m] = forth;
    ++m;
  };
  nlist::visitDirected(cells, i, static_cast<std::int64_t>(i) + 1, take);
  std::sort(dest, dest + m, [](const Bond &a, const Bond &b) { return a.j < b.j; });
  return m;
}

struct WideScratch {
  std::vector<Bond> bonds;
  std::vector<int> bond_at;
  std::vector<int> bond_n;
  std::vector<int> energy_at;
  std::vector<double> energy;
  std::vector<int> local_n;
  std::vector<int> touch;
  std::vector<int> cursor;
  std::vector<Hit> csr;
  std::vector<int> strain_at;
  std::vector<int> strain_n;
  std::vector<int> strain_used;
  std::vector<StrainHit> strain_hit;
};

} // namespace

void TersoffSiC::runOwned(const int *kind, int n, int nt, double cut2, double scale, double *energy,
                         double *acc, double *strain) const {
  thread_local WideScratch scratch;
  auto &bonds = scratch.bonds;
  auto &bond_at = scratch.bond_at;
  auto &bond_n = scratch.bond_n;
  auto &energy_at = scratch.energy_at;
  auto &eterm = scratch.energy;
  auto &strain_at = scratch.strain_at;
  auto &strain_n = scratch.strain_n;
  auto &strain_used = scratch.strain_used;
  auto &strain_hit = scratch.strain_hit;
  const auto &row = neighbours_.offsets();
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
    RadialMemo radial_memo;
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      bond_n[static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    if (t == 0) {
      for (int i = 0; i < n; ++i) {
        bond_at[static_cast<std::size_t>(i) + 1] =
            bond_at[static_cast<std::size_t>(i)] + bond_n[static_cast<std::size_t>(i)];
      }
      if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
        bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const int room =
          bond_at[static_cast<std::size_t>(i) + 1] - bond_at[static_cast<std::size_t>(i)];
      Bond *dest = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      bond_n[static_cast<std::size_t>(i)] =
          storeBonds(neighbours_, m_pair, kind, i, cut2, dest, room, radial_memo);
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    if (t == 0) {
      energy_at.resize(static_cast<std::size_t>(n) + 1);
      energy_at[0] = 0;
      for (int i = 0; i < n; ++i) {
        energy_at[static_cast<std::size_t>(i) + 1] =
            energy_at[static_cast<std::size_t>(i)] + bond_n[static_cast<std::size_t>(i)];
      }
      const int nenergy = energy_at[static_cast<std::size_t>(n)];
      if (static_cast<int>(eterm.size()) < nenergy) {
        eterm.resize(static_cast<std::size_t>(nenergy));
      }
      if (strain != nullptr) {
        strain_at.resize(static_cast<std::size_t>(n) + 1);
        strain_n.assign(static_cast<std::size_t>(n), 0);
        strain_used.assign(static_cast<std::size_t>(n), 0);
        strain_at[0] = 0;
        for (int i = 0; i < n; ++i) {
          const int m = bond_n[static_cast<std::size_t>(i)];
          strain_n[static_cast<std::size_t>(i)] = m + 2 * m * (m > 0 ? m - 1 : 0);
          strain_at[static_cast<std::size_t>(i) + 1] =
              strain_at[static_cast<std::size_t>(i)] + strain_n[static_cast<std::size_t>(i)];
        }
        if (static_cast<int>(strain_hit.size()) < strain_at[static_cast<std::size_t>(n)]) {
          strain_hit.resize(static_cast<std::size_t>(strain_at[static_cast<std::size_t>(n)]));
        }
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    // Each thread owns a contiguous atom range and adds, in center order,
    // every term that lands on those atoms. The bonds are only read.
    std::vector<Angle> ang;
    OrderMemo order_memo;
    const int begin = static_cast<int>(range.begin);
    const int end = static_cast<int>(range.end);
    auto owned = [&](int atom) { return atom >= begin && atom < end; };
    for (int c = 0; c < n; ++c) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(c)];
      const int m = bond_n[static_cast<std::size_t>(c)];
      if (m == 0) {
        continue;
      }
      bool hits = owned(c);
      if (!hits) {
        for (int a = 0; a < m; ++a) {
          if (owned(nb[a].j)) {
            hits = true;
            break;
          }
        }
      }
      if (!hits) {
        continue;
      }
      const TersoffSiC::Species &sp = m_species[kind[c]];
      if (static_cast<int>(ang.size()) < m * m) {
        ang.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
      }
      fillAngles(nb, m, sp, ang.data());
      const bool mine = owned(c);
      double *steps = mine ? eterm.data() + energy_at[static_cast<std::size_t>(c)] : nullptr;
      StrainHit *s_steps =
          mine && strain != nullptr ? strain_hit.data() + strain_at[static_cast<std::size_t>(c)]
                                    : nullptr;
      int ns = 0;
      for (int a = 0; a < m; ++a) {
        const Bond &bj = nb[a];
        double zeta = 0.0;
        for (int k = 0; k < m; ++k) {
          if (k == a) {
            continue;
          }
          const Angle &slot = ang[static_cast<std::size_t>(a * m + k)];
          zeta += nb[k].fc * slot.g;
        }
        double bij = 0.0;
        double db = 0.0;
        order_memo.apply(sp, zeta, bij, db);
        if (steps != nullptr) {
          steps[a] = 0.5 * bj.fc * (bj.f_r + bij * bj.f_a);
        }
        const double dedr =
            0.5 * (bj.dfc * (bj.f_r + bij * bj.f_a) + bj.fc * (bj.df_r + bij * bj.df_a));
        const double inv_r = 1.0 / bj.r;
        const double sx = scale * dedr * bj.x * inv_r;
        const double sy = scale * dedr * bj.y * inv_r;
        const double sz = scale * dedr * bj.z * inv_r;
        if (mine) {
          acc[3 * c] += sx;
          acc[3 * c + 1] += sy;
          acc[3 * c + 2] += sz;
          if (s_steps != nullptr) {
            s_steps[ns] = StrainHit{0, -dedr * inv_r, bj.x, bj.y, bj.z, 0, 0};
            ++ns;
          }
        }
        if (owned(bj.j)) {
          acc[3 * bj.j] -= sx;
          acc[3 * bj.j + 1] -= sy;
          acc[3 * bj.j + 2] -= sz;
        }
        const double dedz = 0.5 * bj.fc * bj.f_a * db;
        if (dedz == 0.0) {
          continue;
        }
        for (int k = 0; k < m; ++k) {
          if (k == a) {
            continue;
          }
          const Bond &bk = nb[k];
          const Angle &slot = ang[static_cast<std::size_t>(a * m + k)];
          const double cth = slot.cth;
          const double g = slot.g;
          const double dg = slot.dg;
          const double inv_r1 = inv_r;
          const double inv_r2 = 1.0 / bk.r;
          const double inv = inv_r1 * inv_r2;
          const double d_edcth = dedz * bk.fc * dg;
          const double d_edr2 = dedz * bk.dfc * g;
          const double g1x = d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
          const double g1y = d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
          const double g1z = d_edcth * (bk.z * inv - cth * bj.z * inv_r1 * inv_r1);
          const double g2x =
              d_edr2 * bk.x * inv_r2 + d_edcth * (bj.x * inv - cth * bk.x * inv_r2 * inv_r2);
          const double g2y =
              d_edr2 * bk.y * inv_r2 + d_edcth * (bj.y * inv - cth * bk.y * inv_r2 * inv_r2);
          const double g2z =
              d_edr2 * bk.z * inv_r2 + d_edcth * (bj.z * inv - cth * bk.z * inv_r2 * inv_r2);
          if (owned(bj.j)) {
            acc[3 * bj.j] -= scale * g1x;
            acc[3 * bj.j + 1] -= scale * g1y;
            acc[3 * bj.j + 2] -= scale * g1z;
          }
          if (owned(bk.j)) {
            acc[3 * bk.j] -= scale * g2x;
            acc[3 * bk.j + 1] -= scale * g2y;
            acc[3 * bk.j + 2] -= scale * g2z;
          }
          if (mine) {
            acc[3 * c] += scale * (g1x + g2x);
            acc[3 * c + 1] += scale * (g1y + g2y);
            acc[3 * c + 2] += scale * (g1z + g2z);
            if (s_steps != nullptr) {
              s_steps[ns] = StrainHit{1, bj.x, bj.y, bj.z, g1x, g1y, g1z};
              ++ns;
              s_steps[ns] = StrainHit{1, bk.x, bk.y, bk.z, g2x, g2y, g2z};
              ++ns;
            }
          }
        }
      }
      if (mine && strain != nullptr) {
        strain_used[static_cast<std::size_t>(c)] = ns;
      }
    }
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
    const int ne = bond_n[static_cast<std::size_t>(i)];
    for (int k = 0; k < ne; ++k) {
      total += steps[k];
    }
  }
  *energy = total;
  if (strain != nullptr) {
    for (int i = 0; i < n; ++i) {
      const StrainHit *hit = strain_hit.data() + strain_at[static_cast<std::size_t>(i)];
      const int ns = strain_used[static_cast<std::size_t>(i)];
      for (int k = 0; k < ns; ++k) {
        if (hit[k].kind == 0) {
          accumulatePairStrain(strain, hit[k].p0, hit[k].p1, hit[k].p2, hit[k].p3);
        } else {
          accumulateBondStrain(strain, hit[k].p0, hit[k].p1, hit[k].p2, hit[k].p3, hit[k].p4,
                               hit[k].p5);
        }
      }
    }
  }
}


void TersoffSiC::runWide(const int *kind, int n, int nt, double cut2, double scale, double *energy,
                         double *acc, double *strain) const {
  // Past 4096 atoms the row fill is slower than each thread adding the
  // terms that land on its own atoms. Smaller lists keep the counted rows.
  if (n >= 4096) {
    runOwned(kind, n, nt, cut2, scale, energy, acc, strain);
    return;
  }
  thread_local WideScratch scratch;
  auto &bonds = scratch.bonds;
  auto &bond_at = scratch.bond_at;
  auto &bond_n = scratch.bond_n;
  auto &energy_at = scratch.energy_at;
  auto &eterm = scratch.energy;
  auto &local_n = scratch.local_n;
  auto &touch = scratch.touch;
  auto &cursor = scratch.cursor;
  auto &csr = scratch.csr;
  auto &strain_at = scratch.strain_at;
  auto &strain_n = scratch.strain_n;
  auto &strain_used = scratch.strain_used;
  auto &strain_hit = scratch.strain_hit;
  const auto &row = neighbours_.offsets();
  bond_at.resize(static_cast<std::size_t>(n) + 1);
  bond_n.assign(static_cast<std::size_t>(n), 0);
  bond_at[0] = 0;
  if (static_cast<int>(local_n.size()) < nt * n) {
    local_n.resize(static_cast<std::size_t>(nt) * static_cast<std::size_t>(n));
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
    RadialMemo radial_memo;
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      bond_n[static_cast<std::size_t>(i)] =
          row[static_cast<std::size_t>(i) + 1] - row[static_cast<std::size_t>(i)];
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    if (t == 0) {
      for (int i = 0; i < n; ++i) {
        bond_at[static_cast<std::size_t>(i) + 1] =
            bond_at[static_cast<std::size_t>(i)] + bond_n[static_cast<std::size_t>(i)];
      }
      if (static_cast<int>(bonds.size()) < bond_at[static_cast<std::size_t>(n)]) {
        bonds.resize(static_cast<std::size_t>(bond_at[static_cast<std::size_t>(n)]));
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const int room =
          bond_at[static_cast<std::size_t>(i) + 1] - bond_at[static_cast<std::size_t>(i)];
      Bond *dest = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      bond_n[static_cast<std::size_t>(i)] =
          storeBonds(neighbours_, m_pair, kind, i, cut2, dest, room, radial_memo);
    }
    int *mine = local_n.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(n);
    std::fill(mine, mine + n, 0);
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const int m = bond_n[static_cast<std::size_t>(i)];
      for (int a = 0; a < m; ++a) {
        ++mine[i];
        ++mine[nb[a].j];
        for (int k = 0; k < m; ++k) {
          if (k == a) {
            continue;
          }
          ++mine[nb[a].j];
          ++mine[nb[k].j];
          ++mine[i];
        }
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
    if (t == 0) {
      touch.resize(static_cast<std::size_t>(n) + 1);
      touch[0] = 0;
      energy_at.resize(static_cast<std::size_t>(n) + 1);
      energy_at[0] = 0;
      for (int i = 0; i < n; ++i) {
        const int m = bond_n[static_cast<std::size_t>(i)];
        energy_at[static_cast<std::size_t>(i) + 1] = energy_at[static_cast<std::size_t>(i)] + m;
        int hits = 0;
        for (int u = 0; u < nt; ++u) {
          hits += local_n[static_cast<std::size_t>(u) * static_cast<std::size_t>(n) +
                          static_cast<std::size_t>(i)];
        }
        touch[static_cast<std::size_t>(i) + 1] = touch[static_cast<std::size_t>(i)] + hits;
      }
      const int nenergy = energy_at[static_cast<std::size_t>(n)];
      const int ntouch = touch[static_cast<std::size_t>(n)];
      if (static_cast<int>(eterm.size()) < nenergy) {
        eterm.resize(static_cast<std::size_t>(nenergy));
      }
      if (static_cast<int>(csr.size()) < ntouch) {
        csr.resize(static_cast<std::size_t>(ntouch));
      }
      if (static_cast<int>(cursor.size()) < nt * n) {
        cursor.resize(static_cast<std::size_t>(nt) * static_cast<std::size_t>(n));
      }
      if (strain != nullptr) {
        strain_at.resize(static_cast<std::size_t>(n) + 1);
        strain_n.assign(static_cast<std::size_t>(n), 0);
        strain_used.assign(static_cast<std::size_t>(n), 0);
        strain_at[0] = 0;
        for (int i = 0; i < n; ++i) {
          const int m = bond_n[static_cast<std::size_t>(i)];
          strain_n[static_cast<std::size_t>(i)] = m + 2 * m * (m - 1);
          strain_at[static_cast<std::size_t>(i) + 1] =
              strain_at[static_cast<std::size_t>(i)] + strain_n[static_cast<std::size_t>(i)];
        }
        if (static_cast<int>(strain_hit.size()) < strain_at[static_cast<std::size_t>(n)]) {
          strain_hit.resize(static_cast<std::size_t>(strain_at[static_cast<std::size_t>(n)]));
        }
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
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
    std::vector<Angle> ang;
    OrderMemo order_memo;
    for (int i = static_cast<int>(range.begin); i < static_cast<int>(range.end); ++i) {
      const Bond *nb = bonds.data() + bond_at[static_cast<std::size_t>(i)];
      const TersoffSiC::Species &sp = m_species[kind[i]];
      const int m = bond_n[static_cast<std::size_t>(i)];
      if (static_cast<int>(ang.size()) < m * m) {
        ang.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(m));
      }
      double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
      StrainHit *s_steps = nullptr;
      if (strain != nullptr) {
        s_steps = strain_hit.data() + strain_at[static_cast<std::size_t>(i)];
      }
      fillAngles(nb, m, sp, ang.data());
      int ns = 0;
      for (int a = 0; a < m; ++a) {
        const Bond &bj = nb[a];
        double zeta = 0.0;
        for (int k = 0; k < m; ++k) {
          if (k == a) {
            continue;
          }
          const Angle &slot = ang[static_cast<std::size_t>(a * m + k)];
          zeta += nb[k].fc * slot.g;
        }
        double bij = 0.0;
        double db = 0.0;
        order_memo.apply(sp, zeta, bij, db);
        steps[a] = 0.5 * bj.fc * (bj.f_r + bij * bj.f_a);
        const double dedr =
            0.5 * (bj.dfc * (bj.f_r + bij * bj.f_a) + bj.fc * (bj.df_r + bij * bj.df_a));
        const double inv_r = 1.0 / bj.r;
        const double sx = scale * dedr * bj.x * inv_r;
        const double sy = scale * dedr * bj.y * inv_r;
        const double sz = scale * dedr * bj.z * inv_r;
        csr[static_cast<std::size_t>(place[i]++)] = Hit{sx, sy, sz};
        csr[static_cast<std::size_t>(place[bj.j]++)] = Hit{-sx, -sy, -sz};
        if (strain != nullptr) {
          s_steps[ns] = StrainHit{0, -dedr * inv_r, bj.x, bj.y, bj.z, 0, 0};
          ++ns;
        }
        const double dedz = 0.5 * bj.fc * bj.f_a * db;
        for (int k = 0; k < m; ++k) {
          if (k == a) {
            continue;
          }
          if (dedz == 0.0) {
            csr[static_cast<std::size_t>(place[bj.j]++)] = Hit{0.0, 0.0, 0.0};
            csr[static_cast<std::size_t>(place[nb[k].j]++)] = Hit{0.0, 0.0, 0.0};
            csr[static_cast<std::size_t>(place[i]++)] = Hit{0.0, 0.0, 0.0};
            continue;
          }
          const Bond &bk = nb[k];
          const Angle &slot = ang[static_cast<std::size_t>(a * m + k)];
          const double cth = slot.cth;
          const double g = slot.g;
          const double dg = slot.dg;
          const double inv_r1 = inv_r;
          const double inv_r2 = 1.0 / bk.r;
          const double inv = inv_r1 * inv_r2;
          const double d_edcth = dedz * bk.fc * dg;
          const double d_edr2 = dedz * bk.dfc * g;
          const double g1x = d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
          const double g1y = d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
          const double g1z = d_edcth * (bk.z * inv - cth * bj.z * inv_r1 * inv_r1);
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
          if (strain != nullptr) {
            s_steps[ns] = StrainHit{1, bj.x, bj.y, bj.z, g1x, g1y, g1z};
            ++ns;
            s_steps[ns] = StrainHit{1, bk.x, bk.y, bk.z, g2x, g2y, g2z};
            ++ns;
          }
        }
      }
      if (strain != nullptr) {
        strain_used[static_cast<std::size_t>(i)] = ns;
      }
    }
#if defined(_OPENMP)
#pragma omp barrier
#endif
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
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const double *steps = eterm.data() + energy_at[static_cast<std::size_t>(i)];
    const int ne = bond_n[static_cast<std::size_t>(i)];
    for (int k = 0; k < ne; ++k) {
      total += steps[k];
    }
  }
  *energy = total;
  if (strain != nullptr) {
    for (int i = 0; i < n; ++i) {
      const StrainHit *hit = strain_hit.data() + strain_at[static_cast<std::size_t>(i)];
      const int ns = strain_used[static_cast<std::size_t>(i)];
      for (int k = 0; k < ns; ++k) {
        if (hit[k].kind == 0) {
          accumulatePairStrain(strain, hit[k].p0, hit[k].p1, hit[k].p2, hit[k].p3);
        } else {
          accumulateBondStrain(strain, hit[k].p0, hit[k].p1, hit[k].p2, hit[k].p3, hit[k].p4,
                               hit[k].p5);
        }
      }
    }
  }
}

} // namespace rgpot
