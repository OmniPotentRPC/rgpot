// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Unshifted cluster pair kernels.
 *
 * The unit Lennard-Jones branch copies the expression tree of anneal's
 * reduced 12-6 pair: inverse powers of r^2, energy 4 (r^-12 - r^-6),
 * and (dV/dr)/r = -24 r^-2 (2 r^-12 - r^-6). Morse at epsilon = 1 and
 * re = 1 copies exp(rho (1 - r)) (exp(...) - 2).
 */

// clang-format off
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
// clang-format on

#include "rgpot/ReducedPair/ReducedPair.hpp"
#include "rgpot/nlist/NeighbourList.hpp"

namespace {

bool bad_cluster(int32_t natoms, const double *positions, double *energy,
                 double *gradient) {
  return natoms < 0 || energy == nullptr ||
         (natoms > 0 && (positions == nullptr || gradient == nullptr));
}

void zero_gradient(int32_t natoms, double *gradient) {
  for (int32_t i = 0; i < 3 * natoms; ++i) {
    gradient[i] = 0.0;
  }
}

bool inside(double r2, double cutoff) {
  if (!(r2 > 0.0)) {
    return false;
  }
  if (!(cutoff > 0.0)) {
    return true;
  }
  const double cut2 = cutoff * cutoff;
  return r2 < cut2;
}

struct ClusterWalk {
  rgpot::nlist::Mode mode = rgpot::nlist::Mode::Stored;
  rgpot::nlist::Neighbours cells;
  std::vector<double> shifted;
  std::size_t slots = 0;
};

ClusterWalk &clusterWalk() {
  thread_local ClusterWalk walk;
  return walk;
}

void bindCluster(const double *pos, int32_t n, double cutoff) {
  auto &walk = clusterWalk();
  walk.shifted.assign(static_cast<std::size_t>(std::max(n, 0)) * 3, 0.0);
  double search = 1.0;
  double box[9] = {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0};
  if (n > 0 && pos != nullptr) {
    double minv[3] = {pos[0], pos[1], pos[2]};
    double maxv[3] = {minv[0], minv[1], minv[2]};
    for (int32_t i = 1; i < n; ++i) {
      for (int k = 0; k < 3; ++k) {
        const double x = pos[3 * i + k];
        minv[k] = std::min(minv[k], x);
        maxv[k] = std::max(maxv[k], x);
      }
    }
    double span2 = 0.0;
    for (int k = 0; k < 3; ++k) {
      const double span = maxv[k] - minv[k];
      span2 += span * span;
    }
    if (cutoff > 0.0) {
      search = cutoff;
    } else {
      search = std::sqrt(span2) * (1.0 + 1.0e-9) + 1.0e-6;
      if (!(search > 0.0)) {
        search = 1.0;
      }
    }
    constexpr double pad = 1.0;
    for (int k = 0; k < 3; ++k) {
      const double span = maxv[k] - minv[k];
      double length = span + 2.0 * pad;
      if (length < search + 2.0 * pad) {
        length = search + 2.0 * pad;
      }
      box[4 * k] = length;
      const double origin = minv[k] - pad;
      for (int32_t i = 0; i < n; ++i) {
        walk.shifted[static_cast<std::size_t>(3 * i + k)] = pos[3 * i + k] - origin;
      }
    }
  }
  const double layerCut = cutoff > 0.0 ? cutoff : search;
  const double skin =
      (walk.mode == rgpot::nlist::Mode::Stream || !(cutoff > 0.0)) ? 0.0 : cutoff / 5.0;
  rgpot::nlist::updateCells(walk.cells, walk.shifted.data(), static_cast<std::size_t>(std::max(n, 0)),
                            box, layerCut, skin, rgpot::nlist::Image::Round, walk.mode,
                            {{false, false, false}});
  walk.slots = walk.cells.streaming() ? 0 : walk.cells.slots();
}

template <typename Fn>
void visitHalf(const double *pos, int32_t n, double cutoff, Fn &&fn) {
  bindCluster(pos, n, cutoff);
  if (n < 2) {
    return;
  }
  rgpot::nlist::visitHalf(
      clusterWalk().cells, 0, n,
      [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
        if (!inside(r2, cutoff)) {
          return;
        }
        fn(i, j, dx, dy, dz, r2);
      });
}

void lj_unit(double r2, double *energy, double *coef, double *curv) {
  const double inv2 = 1.0 / r2;
  const double inv6 = inv2 * inv2 * inv2;
  const double inv12 = inv6 * inv6;
  *energy = 4.0 * (inv12 - inv6);
  *coef = -24.0 * inv2 * (2.0 * inv12 - inv6);
  if (curv != nullptr) {
    *curv = inv2 * (624.0 * inv12 - 168.0 * inv6);
  }
}

void lj_general(double epsilon, double sigma, double r2, double *energy,
                double *coef, double *curv) {
  const double inv2 = 1.0 / r2;
  const double scaled = (sigma * sigma) * inv2;
  const double s6 = scaled * scaled * scaled;
  const double s12 = s6 * s6;
  *energy = 4.0 * epsilon * (s12 - s6);
  *coef = -24.0 * epsilon * inv2 * (2.0 * s12 - s6);
  if (curv != nullptr) {
    *curv = epsilon * inv2 * (624.0 * s12 - 168.0 * s6);
  }
}

void lj_terms(double epsilon, double sigma, double r2, double *energy,
              double *coef, double *curv) {
  if (!(r2 > 0.0) || !std::isfinite(r2)) {
    *energy = 0.0;
    *coef = 0.0;
    if (curv != nullptr) {
      *curv = 0.0;
    }
    return;
  }
  if (epsilon == 1.0 && sigma == 1.0) {
    lj_unit(r2, energy, coef, curv);
    return;
  }
  lj_general(epsilon, sigma, r2, energy, coef, curv);
}

void add_pair(double *gradient, int32_t i, int32_t j, const double d[3],
              double coef) {
  for (int k = 0; k < 3; ++k) {
    const double term = coef * d[k];
    gradient[3 * i + k] += term;
    gradient[3 * j + k] -= term;
  }
}

void morse_unit(double rho, double r2, double *energy, double *coef,
                double *curv) {
  const double r = std::sqrt(r2);
  const double a = std::exp(rho * (1.0 - r));
  *energy = a * (a - 2.0);
  *coef = 2.0 * rho * (a - a * a) / r;
  if (curv != nullptr) {
    *curv = 2.0 * rho * rho * (2.0 * a * a - a);
  }
}

void morse_general(double epsilon, double rho, double re, double r2,
                   double *energy, double *coef, double *curv) {
  const double r = std::sqrt(r2);
  const double b = std::exp(rho * (1.0 - r / re));
  *energy = epsilon * b * (b - 2.0);
  *coef = epsilon * 2.0 * rho / re * b * (1.0 - b) / r;
  if (curv != nullptr) {
    const double db = -b * rho / re;
    const double d2b = b * (rho / re) * (rho / re);
    *curv = epsilon * (2.0 * db * db + 2.0 * (b - 1.0) * d2b);
  }
}

} // namespace

extern "C" RGPOT_PAIR_API int rgpot_lj_pair(double epsilon, double sigma,
                                            double r2, double *energy,
                                            double *dvdr_over_r,
                                            double *d2vdr2) {
  if (energy == nullptr || dvdr_over_r == nullptr || d2vdr2 == nullptr) {
    return 1;
  }
  lj_terms(epsilon, sigma, r2, energy, dvdr_over_r, d2vdr2);
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_lj_curvature_pairs(double epsilon, double sigma, int32_t npairs,
                         const double *r2, double *energy, double *dvdr_over_r,
                         double *d2vdr2) {
  if (npairs < 0 ||
      (npairs > 0 && (r2 == nullptr || energy == nullptr ||
                      dvdr_over_r == nullptr || d2vdr2 == nullptr))) {
    return 1;
  }
  for (int32_t i = 0; i < npairs; ++i) {
    lj_terms(epsilon, sigma, r2[i], &energy[i], &dvdr_over_r[i], &d2vdr2[i]);
  }
  return 0;
}

extern "C" RGPOT_PAIR_API void rgpot_pair_set_neighbour_mode(int32_t mode) {
  auto &walk = clusterWalk();
  if (mode == 2) {
    walk.mode = rgpot::nlist::Mode::Stream;
  } else if (mode == 0) {
    walk.mode = rgpot::nlist::Mode::Auto;
  } else {
    walk.mode = rgpot::nlist::Mode::Stored;
  }
}

extern "C" RGPOT_PAIR_API int64_t rgpot_pair_neighbour_slots(void) {
  return static_cast<int64_t>(clusterWalk().slots);
}

extern "C" RGPOT_PAIR_API int
rgpot_lj_gradient(double epsilon, double sigma, double cutoff, int32_t natoms,
                  const double *positions, double *energy, double *gradient) {
  if (bad_cluster(natoms, positions, energy, gradient)) {
    return 1;
  }
  zero_gradient(natoms, gradient);
  double e = 0.0;
  visitHalf(positions, natoms, cutoff,
            [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
              double d[3] = {dx, dy, dz};
              double v = 0.0;
              double coef = 0.0;
              lj_terms(epsilon, sigma, r2, &v, &coef, nullptr);
              e += v;
              add_pair(gradient, i, j, d, coef);
            });
  *energy = e;
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_lj_atom_energies(double epsilon, double sigma, int32_t natoms,
                       const double *positions, double *atom_energy) {
  if (natoms < 0 || atom_energy == nullptr ||
      (natoms > 0 && positions == nullptr)) {
    return 1;
  }
  for (int32_t i = 0; i < natoms; ++i) {
    atom_energy[i] = 0.0;
  }
  visitHalf(positions, natoms, 0.0,
            [&](std::int32_t i, std::int32_t j, double, double, double, double r2) {
              const double scaled = (sigma * sigma) / r2;
              const double s6 = scaled * scaled * scaled;
              const double v = 4.0 * epsilon * (s6 * s6 - s6);
              atom_energy[i] += v;
              atom_energy[j] += v;
            });
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_lj_partial_gradient(double epsilon, double sigma, int32_t natoms,
                          const double *positions, int32_t n_moved,
                          const int32_t *moved, double *gradient) {
  if (natoms < 0 || gradient == nullptr || n_moved < 0 ||
      (natoms > 0 && positions == nullptr) ||
      (n_moved > 0 && moved == nullptr)) {
    return 1;
  }
  zero_gradient(natoms, gradient);
  for (int32_t m = 0; m < n_moved; ++m) {
    const int32_t i = moved[m];
    if (i < 0 || i >= natoms) {
      return 1;
    }
  }
  bindCluster(positions, natoms, 0.0);
  if (natoms < 2) {
    return 0;
  }
  auto &walk = clusterWalk();
  for (int32_t m = 0; m < n_moved; ++m) {
    const int32_t i = moved[m];
    walk.cells.forAtom(i, [&](std::int32_t, std::int32_t, double dx, double dy, double dz,
                              double r2) {
      if (!(r2 > 0.0)) {
        return;
      }
      double v = 0.0;
      double coef = 0.0;
      lj_terms(epsilon, sigma, r2, &v, &coef, nullptr);
      const double d[3] = {dx, dy, dz};
      for (int k = 0; k < 3; ++k) {
        gradient[3 * i + k] += coef * d[k];
      }
    });
  }
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_lj_arithmetic_gradient(const double *sigma, double cutoff, int32_t natoms,
                             const double *positions, double *energy,
                             double *gradient) {
  if (bad_cluster(natoms, positions, energy, gradient) ||
      (natoms > 0 && sigma == nullptr)) {
    return 1;
  }
  zero_gradient(natoms, gradient);
  double e = 0.0;
  visitHalf(positions, natoms, cutoff,
            [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
              double d[3] = {dx, dy, dz};
              const double sum = sigma[i] + sigma[j];
              const double s2 = 0.25 * sum * sum;
              const double inv = 1.0 / r2;
              const double sr2 = s2 * inv;
              const double sr6 = sr2 * sr2 * sr2;
              const double sr12 = sr6 * sr6;
              e += sr12 - sr6;
              const double slope = 24.0 * (sr6 - 2.0 * sr12) * inv;
              add_pair(gradient, i, j, d, slope);
            });
  *energy = 4.0 * e;
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_morse_pair(double epsilon, double rho, double re, double r2,
                 double *energy, double *dvdr_over_r, double *d2vdr2) {
  if (energy == nullptr || dvdr_over_r == nullptr || d2vdr2 == nullptr) {
    return 1;
  }
  if (!(r2 > 0.0) || !(re > 0.0)) {
    *energy = 0.0;
    *dvdr_over_r = 0.0;
    *d2vdr2 = 0.0;
    return 0;
  }
  if (epsilon == 1.0 && re == 1.0) {
    morse_unit(rho, r2, energy, dvdr_over_r, d2vdr2);
    return 0;
  }
  morse_general(epsilon, rho, re, r2, energy, dvdr_over_r, d2vdr2);
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_morse_gradient(double epsilon, double rho, double re, double cutoff,
                     int32_t natoms, const double *positions, double *energy,
                     double *gradient) {
  if (bad_cluster(natoms, positions, energy, gradient) || !(re > 0.0)) {
    return 1;
  }
  zero_gradient(natoms, gradient);
  double e = 0.0;
  visitHalf(positions, natoms, cutoff,
            [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
              double d[3] = {dx, dy, dz};
              double v = 0.0;
              double coef = 0.0;
              double curv = 0.0;
              rgpot_morse_pair(epsilon, rho, re, r2, &v, &coef, &curv);
              e += v;
              add_pair(gradient, i, j, d, coef);
            });
  *energy = e;
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_binary_lj_gradient(int32_t n_a, const double *eps, const double *sig,
                         double cutoff, int32_t natoms, const double *positions,
                         double *energy, double *gradient) {
  if (bad_cluster(natoms, positions, energy, gradient) || n_a < 0 ||
      n_a > natoms || eps == nullptr || sig == nullptr) {
    return 1;
  }
  if (eps[0] == 1.0 && eps[1] == 1.0 && eps[2] == 1.0 && sig[0] == 1.0 &&
      sig[1] == 1.0 && sig[2] == 1.0 && !(cutoff > 0.0)) {
    return rgpot_lj_gradient(1.0, 1.0, 0.0, natoms, positions, energy,
                             gradient);
  }
  zero_gradient(natoms, gradient);
  double e = 0.0;
  visitHalf(positions, natoms, cutoff,
            [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz, double r2) {
              double d[3] = {dx, dy, dz};
              const int kind = (i < n_a ? 0 : 2) + (j < n_a ? 0 : 1);
              const int index = kind == 0 ? 0 : (kind == 3 ? 2 : 1);
              const double a = (sig[index] * sig[index]) / r2;
              const double a3 = a * a * a;
              const double a6 = a3 * a3;
              e += 4.0 * eps[index] * (a6 - a3);
              const double coef = 4.0 * eps[index] * (-12.0 * a6 + 6.0 * a3) / r2;
              add_pair(gradient, i, j, d, coef);
            });
  *energy = e;
  return 0;
}

extern "C" RGPOT_PAIR_API int
rgpot_tip4p_sites(const rgpot_tip4p_sites_config *cfg, int32_t n_molecules,
                  const double *sites, double *energy, double *gradient) {
  if (cfg == nullptr || n_molecules < 0 || energy == nullptr ||
      (n_molecules > 0 && (sites == nullptr || gradient == nullptr))) {
    return 1;
  }
  const int32_t ncoord = 12 * n_molecules;
  for (int32_t i = 0; i < ncoord; ++i) {
    gradient[i] = 0.0;
  }
  const double charges[4] = {0.0, cfg->q_h, cfg->q_h, cfg->q_m};
  std::vector<double> oxygen(static_cast<std::size_t>(std::max(n_molecules, 0)) * 3, 0.0);
  for (int32_t m = 0; m < n_molecules; ++m) {
    oxygen[static_cast<std::size_t>(3 * m)] = sites[12 * m];
    oxygen[static_cast<std::size_t>(3 * m + 1)] = sites[12 * m + 1];
    oxygen[static_cast<std::size_t>(3 * m + 2)] = sites[12 * m + 2];
  }
  double e = 0.0;
  visitHalf(oxygen.data(), n_molecules, 0.0,
            [&](std::int32_t left, std::int32_t right, double dx, double dy, double dz, double r2) {
      const double d[3] = {dx, dy, dz};
      if (r2 > 0.0) {
        double coef = 0.0;
        if (cfg->use_epsilon) {
          const double inv_r2 = 1.0 / r2;
          const double scaled = (cfg->sigma * cfg->sigma) * inv_r2;
          const double inv6 = scaled * scaled * scaled;
          const double inv12 = inv6 * inv6;
          e += 4.0 * cfg->epsilon * (inv12 - inv6);
          coef = -24.0 * cfg->epsilon * inv_r2 * (2.0 * inv12 - inv6);
        } else {
          const double inverse = 1.0 / r2;
          const double r6 = inverse * inverse * inverse;
          const double r12 = r6 * r6;
          e += cfg->c12 * r12 - cfg->c6 * r6;
          coef = (-12.0 * cfg->c12 * r12 + 6.0 * cfg->c6 * r6) * inverse;
        }
        for (int k = 0; k < 3; ++k) {
          gradient[12 * left + k] += coef * d[k];
          gradient[12 * right + k] -= coef * d[k];
        }
      }
      for (int a = 1; a < 4; ++a) {
        for (int b = 1; b < 4; ++b) {
          double cd[3];
          double cr2 = 0.0;
          for (int k = 0; k < 3; ++k) {
            cd[k] =
                sites[12 * left + 3 * a + k] - sites[12 * right + 3 * b + k];
            cr2 += cd[k] * cd[k];
          }
          if (!(cr2 > 0.0)) {
            continue;
          }
          const double distance = std::sqrt(cr2);
          double pair = 0.0;
          double slope = 0.0;
          if (cfg->use_epsilon) {
            const double qq = cfg->coulomb * charges[a] * charges[b];
            pair = qq / distance;
            slope = -qq / (cr2 * distance);
          } else {
            pair = cfg->coulomb * charges[a] * charges[b] / distance;
            slope = -pair / cr2;
          }
          e += pair;
          for (int k = 0; k < 3; ++k) {
            gradient[12 * left + 3 * a + k] += slope * cd[k];
            gradient[12 * right + 3 * b + k] -= slope * cd[k];
          }
        }
      }
    });
  *energy = e;
  return 0;
}
