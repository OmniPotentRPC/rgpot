// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file TersoffSiC.cc
 * @author rgpot Developers
 * @date 2026-10-26
 * @brief Tersoff Si-C kernel. lambda3 is zero for this parameter set, so
 * the coordination factor exp(lambda3^m (rij-rik)^m) is omitted.
 */

#include "rgpot/Tersoff/TersoffSiC.hpp"

#include "rgpot/ParamHash.hpp"
#include "rgpot/nlist/MinimumImage.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Homoatomic columns. R and D in the LAMMPS file are the taper midpoint
// and half-width; Rin = R - D and Rout = R + D.
struct Homo {
  double big_a, big_b, lambda1, lambda2, beta, n, c, d, h, rin, rout, gamma;
};

constexpr Homo kSi = {1830.8, 471.18, 2.4799,   1.7322, 1.1e-6, 0.78734,
                      100390, 16.217, -0.59825, 2.7,    3.0,    1.0};
constexpr Homo kC = {1393.6, 346.74, 3.4879,   2.2119, 1.5724e-7, 0.72751,
                     38049,  4.3484, -0.57058, 1.8,    2.1,       1.0};
constexpr double kMaxCut = kSi.rout;

struct Bond {
  int j;
  double x, y, z, r;
  double fc, dfc, f_r, f_a, df_r, df_a;
};

struct Scratch {
  std::vector<std::vector<Bond>> rows;
  std::vector<int> kind;
};

Scratch &scratch() {
  thread_local Scratch s;
  return s;
}

int kindOf(int z) {
  if (z == 6) {
    return 0;
  }
  if (z == 14) {
    return 1;
  }
  throw std::invalid_argument("TersoffSiC: atomic number " + std::to_string(z) +
                              " is neither carbon (6) nor silicon (14)");
}

TersoffSiC::Species makeSpecies(const Homo &h) {
  TersoffSiC::Species s;
  s.c = h.c;
  s.d = h.d;
  s.h = h.h;
  s.n = h.n;
  s.beta = h.beta;
  s.gamma = h.gamma;
  s.c2 = h.c * h.c;
  s.d2 = h.d * h.d;
  s.beta_n = std::pow(h.beta, h.n);
  return s;
}

TersoffSiC::Pair mixPair(const Homo &a, const Homo &b, double chi) {
  TersoffSiC::Pair p;
  p.lambda1 = 0.5 * (a.lambda1 + b.lambda1);
  p.lambda2 = 0.5 * (a.lambda2 + b.lambda2);
  p.big_a = std::sqrt(a.big_a * b.big_a);
  p.big_b = chi * std::sqrt(a.big_b * b.big_b);
  // Inner and outer cutoffs are averaged separately. Averaging the
  // LAMMPS midpoints is a different number.
  p.rin = std::sqrt(a.rin * b.rin);
  p.rout = std::sqrt(a.rout * b.rout);
  return p;
}

void angular(const TersoffSiC::Species &sp, double cth, double &g, double &dg) {
  const double gap = sp.h - cth;
  const double den = sp.d2 + gap * gap;
  g = sp.gamma * (1.0 + sp.c2 / sp.d2 - sp.c2 / den);
  dg = sp.gamma * (-2.0 * sp.c2 * gap) / (den * den);
}

void bondOrder(const TersoffSiC::Species &sp, double zeta, double &b,
               double &db) {
  if (zeta == 0.0) {
    b = 1.0;
    db = 0.0;
    return;
  }
  const double base = 1.0 + sp.beta_n * std::pow(zeta, sp.n);
  b = std::pow(base, -0.5 / sp.n);
  db = -0.5 * std::pow(base, -0.5 / sp.n - 1.0) * sp.beta_n *
       std::pow(zeta, sp.n - 1.0);
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

bool orthogonal(const double *box, double &lx, double &ly, double &lz) {
  if (box[1] != 0.0 || box[2] != 0.0 || box[3] != 0.0 || box[5] != 0.0 ||
      box[6] != 0.0 || box[7] != 0.0) {
    return false;
  }
  lx = box[0];
  ly = box[4];
  lz = box[8];
  return lx > 0.0 && ly > 0.0 && lz > 0.0 && std::isfinite(lx) &&
         std::isfinite(ly) && std::isfinite(lz);
}

void foldOrtho(double &dx, double &dy, double &dz, double lx, double ly,
               double lz) {
  dx -= lx * std::round(dx / lx);
  dy -= ly * std::round(dy / ly);
  dz -= lz * std::round(dz / lz);
}

void resetRows(Scratch &s, std::size_t n) {
  if (s.rows.size() < n) {
    s.rows.resize(n);
  }
  for (std::size_t i = 0; i < n; ++i) {
    s.rows[i].clear();
    if (s.rows[i].capacity() < 32) {
      s.rows[i].reserve(32);
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

} // namespace

TersoffSiC::TersoffSiC() : Potential(PotType::TersoffSi) {
  m_species[0] = makeSpecies(kC);
  m_species[1] = makeSpecies(kSi);
  const Homo homo[2] = {kC, kSi};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const double chi = (i == j) ? 1.0 : kChiSiC;
      m_pair[i][j] = mixPair(homo[i], homo[j], chi);
    }
  }
  Fnv1a fp;
  fp.u64(1);
  fp.f64(kChiSiC);
  const Homo both[2] = {kC, kSi};
  for (const Homo &h : both) {
    fp.f64(h.big_a);
    fp.f64(h.big_b);
    fp.f64(h.lambda1);
    fp.f64(h.lambda2);
    fp.f64(h.beta);
    fp.f64(h.n);
    fp.f64(h.c);
    fp.f64(h.d);
    fp.f64(h.h);
    fp.f64(h.rin);
    fp.f64(h.rout);
    fp.f64(h.gamma);
  }
  m_paramsKey = fp.h;
}

void TersoffSiC::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (out == nullptr) {
    throw std::invalid_argument("TersoffSiC: null ForceOut");
  }
  out->variance = 0.0;
  out->has_stress = 0;
  for (double &s : out->stress) {
    s = 0.0;
  }
  if (in.nAtoms > 0 && (in.pos == nullptr || in.box == nullptr ||
                        in.atmnrs == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("TersoffSiC: null force buffer");
  }
  evaluate(in.pos, in.atmnrs, in.box, in.nAtoms, &out->energy, out->F, 1.0,
           nullptr);
}

void TersoffSiC::energyAndGradient(const double *positions,
                                   const int *atomic_numbers, const double *box,
                                   std::size_t n_atoms, double *energy,
                                   double *gradient,
                                   nlist::NeighborList *neighbors) const {
  if (energy == nullptr ||
      (n_atoms > 0 && (positions == nullptr || atomic_numbers == nullptr ||
                       box == nullptr || gradient == nullptr))) {
    throw std::invalid_argument("TersoffSiC: null energy buffer");
  }
  evaluate(positions, atomic_numbers, box, n_atoms, energy, gradient, -1.0,
           neighbors);
}

void TersoffSiC::energyAndGradientBatch(std::size_t n_config,
                                        const double *positions,
                                        const int *atomic_numbers,
                                        const double *box, std::size_t n_atoms,
                                        double *energies,
                                        double *gradients) const {
  if (n_config == 0) {
    return;
  }
  if (energies == nullptr ||
      (n_atoms > 0 && (positions == nullptr || atomic_numbers == nullptr ||
                       box == nullptr || gradients == nullptr))) {
    throw std::invalid_argument("TersoffSiC: null batch buffer");
  }
  const std::size_t stride = n_atoms * 3;
  for (std::size_t c = 0; c < n_config; ++c) {
    energyAndGradient(positions + c * stride, atomic_numbers, box, n_atoms,
                      energies + c, gradients + c * stride);
  }
}

void TersoffSiC::evaluate(const double *positions, const int *atomic_numbers,
                          const double *box, std::size_t n_atoms,
                          double *energy, double *acc, double scale,
                          nlist::NeighborList *neighbors) const {
  *energy = 0.0;
  for (std::size_t i = 0; i < n_atoms * 3; ++i) {
    acc[i] = 0.0;
  }
  if (n_atoms < 2) {
    if (n_atoms == 1) {
      kindOf(atomic_numbers[0]);
    }
    return;
  }

  double lx = 0.0;
  double ly = 0.0;
  double lz = 0.0;
  const bool ortho = orthogonal(box, lx, ly, lz);
  nlist::MinimumImage image;
  if (!ortho) {
    image = nlist::MinimumImage(box, {true, true, true});
  }

  Scratch &sc = scratch();
  resetRows(sc, n_atoms);
  sc.kind.resize(n_atoms);
  for (std::size_t i = 0; i < n_atoms; ++i) {
    sc.kind[i] = kindOf(atomic_numbers[i]);
  }
  const int n = static_cast<int>(n_atoms);
  const double cut2 = kMaxCut * kMaxCut;

  auto consider = [&](int i, int j, double dx, double dy, double dz) {
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (!(r2 > 0.0) || r2 >= cut2) {
      return;
    }
    const double r = std::sqrt(r2);
    const Pair &pij = m_pair[sc.kind[static_cast<std::size_t>(i)]]
                            [sc.kind[static_cast<std::size_t>(j)]];
    if (r >= pij.rout) {
      return;
    }
    Bond forth{j, dx, dy, dz, r, 0, 0, 0, 0, 0, 0};
    radial(pij, r, forth);
    Bond back{i,         -dx,       -dy,       -dz,        r,         forth.fc,
              forth.dfc, forth.f_r, forth.f_a, forth.df_r, forth.df_a};
    sc.rows[static_cast<std::size_t>(i)].push_back(forth);
    sc.rows[static_cast<std::size_t>(j)].push_back(back);
  };

  const nlist::NeighborList &pairs =
      nlist::orderedPairs(positions, box, n_atoms, kMaxCut, neighbors);
  if (pairs.use_stored) {
    for (std::size_t p = 0; p < pairs.lo.size(); ++p) {
      consider(pairs.lo[p], pairs.hi[p], pairs.dx[p], pairs.dy[p], pairs.dz[p]);
    }
  } else {
    for (std::size_t p = 0; p < pairs.lo.size(); ++p) {
      const int i = pairs.lo[p];
      const int j = pairs.hi[p];
      double dx = positions[3 * j] - positions[3 * i];
      double dy = positions[3 * j + 1] - positions[3 * i + 1];
      double dz = positions[3 * j + 2] - positions[3 * i + 2];
      if (ortho) {
        foldOrtho(dx, dy, dz, lx, ly, lz);
      } else {
        image.fold(dx, dy, dz);
      }
      consider(i, j, dx, dy, dz);
    }
  }

  for (int i = 0; i < n; ++i) {
    const Species &sp = m_species[sc.kind[static_cast<std::size_t>(i)]];
    const std::vector<Bond> &nb = sc.rows[static_cast<std::size_t>(i)];
    const int m = static_cast<int>(nb.size());
    for (int a = 0; a < m; ++a) {
      const Bond &bj = nb[static_cast<std::size_t>(a)];
      double zeta = 0.0;
      for (int k = 0; k < m; ++k) {
        if (k == a) {
          continue;
        }
        double g = 0.0;
        double dg = 0.0;
        angular(sp, cosine(bj, nb[static_cast<std::size_t>(k)]), g, dg);
        zeta += nb[static_cast<std::size_t>(k)].fc * g;
      }
      double bij = 0.0;
      double db = 0.0;
      bondOrder(sp, zeta, bij, db);
      *energy += 0.5 * bj.fc * (bj.f_r + bij * bj.f_a);
      const double dedr = 0.5 * (bj.dfc * (bj.f_r + bij * bj.f_a) +
                                 bj.fc * (bj.df_r + bij * bj.df_a));
      const double inv_r = 1.0 / bj.r;
      const double sx = scale * dedr * bj.x * inv_r;
      const double sy = scale * dedr * bj.y * inv_r;
      const double sz = scale * dedr * bj.z * inv_r;
      acc[3 * i] += sx;
      acc[3 * i + 1] += sy;
      acc[3 * i + 2] += sz;
      acc[3 * bj.j] -= sx;
      acc[3 * bj.j + 1] -= sy;
      acc[3 * bj.j + 2] -= sz;
      const double dedz = 0.5 * bj.fc * bj.f_a * db;
      if (dedz == 0.0) {
        continue;
      }
      for (int k = 0; k < m; ++k) {
        if (k == a) {
          continue;
        }
        const Bond &bk = nb[static_cast<std::size_t>(k)];
        const double cth = cosine(bj, bk);
        double g = 0.0;
        double dg = 0.0;
        angular(sp, cth, g, dg);
        const double inv_r1 = inv_r;
        const double inv_r2 = 1.0 / bk.r;
        const double inv = inv_r1 * inv_r2;
        const double d_edcth = dedz * bk.fc * dg;
        const double d_edr2 = dedz * bk.dfc * g;
        const double g1x =
            d_edcth * (bk.x * inv - cth * bj.x * inv_r1 * inv_r1);
        const double g1y =
            d_edcth * (bk.y * inv - cth * bj.y * inv_r1 * inv_r1);
        const double g1z =
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
}

} // namespace rgpot
