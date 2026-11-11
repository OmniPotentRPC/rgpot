// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the ZBL potential methods.
 *
 * Ported from eOn (https://github.com/TheochemUI/eOn,
 * client/potentials/ZBL), BSD-3-Clause licensed, copyright the eOn
 * Development Team, which adapts the GPL-licensed LAMMPS ``pair_zbl``
 * kernel. Constants follow LAMMPS ``metal`` units: eV, Angstrom.
 */

// clang-format off
#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
// clang-format on

#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/simd/from_cache.hpp"
#include "rgpot/stress.hpp"

namespace rgpot {

namespace {

// Universal ZBL screening function coefficients (pair_zbl_const.h).
constexpr double kC1 = 0.02817;
constexpr double kC2 = 0.28022;
constexpr double kC3 = 0.50986;
constexpr double kC4 = 0.18175;

// Screening exponents as carried by the LAMMPS source, which differ in the
// last digits from the published values.
constexpr double kD1 = 3.19980;
constexpr double kD2 = 0.94229;
constexpr double kD3 = 0.40290;
constexpr double kD4 = 0.20162;

constexpr double kPZBL = 0.23;
/// Screening length prefactor (Angstrom).
constexpr double kA0 = 0.46850;
/// Coulomb conversion in LAMMPS 'metal' units (eV Angstrom).
constexpr double kQQR2E = 14.399645;

/// Unswitched screened Coulomb energy.
double e_zbl(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double sum = kC4 * std::exp(-p.d1 * r) + kC3 * std::exp(-p.d2 * r) +
                     kC2 * std::exp(-p.d3 * r) + kC1 * std::exp(-p.d4 * r);
  return p.zze * sum * r_inv;
}

/// First derivative of the unswitched energy with respect to r.
double dzbldr(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double e1 = std::exp(-p.d1 * r);
  const double e2 = std::exp(-p.d2 * r);
  const double e3 = std::exp(-p.d3 * r);
  const double e4 = std::exp(-p.d4 * r);

  const double sum = kC4 * e1 + kC3 * e2 + kC2 * e3 + kC1 * e4;
  const double sum_p =
      -kC4 * p.d1 * e1 - kC3 * p.d2 * e2 - kC2 * p.d3 * e3 - kC1 * p.d4 * e4;

  return p.zze * (sum_p - sum * r_inv) * r_inv;
}

/// Second derivative of the unswitched energy with respect to r.
double d2zbldr2(const ZblPairCoeffs &p, double r) {
  const double r_inv = 1.0 / r;
  const double e1 = std::exp(-p.d1 * r);
  const double e2 = std::exp(-p.d2 * r);
  const double e3 = std::exp(-p.d3 * r);
  const double e4 = std::exp(-p.d4 * r);

  const double sum = kC4 * e1 + kC3 * e2 + kC2 * e3 + kC1 * e4;
  const double sum_p =
      kC4 * e1 * p.d1 + kC3 * e2 * p.d2 + kC2 * e3 * p.d3 + kC1 * e4 * p.d4;
  const double sum_pp = kC4 * e1 * p.d1 * p.d1 + kC3 * e2 * p.d2 * p.d2 +
                        kC2 * e3 * p.d3 * p.d3 + kC1 * e4 * p.d4 * p.d4;

  return p.zze * (sum_pp + 2.0 * sum_p * r_inv + 2.0 * sum * r_inv * r_inv) *
         r_inv;
}

} // namespace

ZBLPot::ZBLPot(const ZBLConfig &c)
    : Potential(PotType::ZBL), cut_inner{c.cut_inner}, cut_global{c.cut_global},
      cut_inner_sq{c.cut_inner * c.cut_inner}, m_config{c} {
  if (!(cut_inner > 0.0) || !(cut_inner < cut_global)) {
    throw std::invalid_argument(
        "ZBLPot: invalid cutoffs, require 0.0 < cut_inner < cut_global.");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.cut_inner);
  fp.f64(c.cut_global);
  m_paramsKey = fp.h;
}

std::shared_ptr<const ZblTables>
ZBLPot::buildTables(std::vector<int> uniqueZ) const {
  auto tables = std::make_shared<ZblTables>();
  tables->z = std::move(uniqueZ);
  const std::size_t n = tables->z.size();
  tables->pairs.assign(n * n, ZblPairCoeffs{});

  const double tc = cut_global - cut_inner;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      const double zi = static_cast<double>(tables->z[i]);
      const double zj = static_cast<double>(tables->z[j]);
      const double a_inv = (std::pow(zi, kPZBL) + std::pow(zj, kPZBL)) / kA0;

      ZblPairCoeffs p;
      p.d1 = kD1 * a_inv;
      p.d2 = kD2 * a_inv;
      p.d3 = kD3 * a_inv;
      p.d4 = kD4 * a_inv;
      p.zze = zi * zj * kQQR2E;

      // Switching polynomial: energy and force both vanish at cut_global.
      const double fc = e_zbl(p, cut_global);
      const double fcp = dzbldr(p, cut_global);
      const double fcpp = d2zbldr2(p, cut_global);
      const double swa = (-3.0 * fcp + tc * fcpp) / (tc * tc);
      const double swb = (2.0 * fcp - tc * fcpp) / (tc * tc * tc);
      const double swc = -fc + (tc / 2.0) * fcp - (tc * tc / 12.0) * fcpp;

      p.sw1 = swa;
      p.sw2 = swb;
      p.sw3 = swa / 3.0;
      p.sw4 = swb / 4.0;
      p.sw5 = swc;

      tables->pairs[i * n + j] = p;
      tables->pairs[j * n + i] = p;
    }
  }
  return tables;
}

std::shared_ptr<const ZblTables> ZBLPot::tablesFor(const int *atomicNrs,
                                                   long N) const {
  std::vector<int> uniqueZ(atomicNrs, atomicNrs + N);
  std::sort(uniqueZ.begin(), uniqueZ.end());
  uniqueZ.erase(std::unique(uniqueZ.begin(), uniqueZ.end()), uniqueZ.end());

  {
    std::lock_guard<std::mutex> lock(m_tablesMtx);
    if (m_tables && m_tables->z == uniqueZ) {
      return m_tables;
    }
  }

  auto fresh = buildTables(std::move(uniqueZ));
  std::lock_guard<std::mutex> lock(m_tablesMtx);
  m_tables = fresh;
  return fresh;
}

/**
 * @class ZBLPot
 * @details
 *
 * Every pair within ``cut_global`` contributes, with vectors taken
 * straight from the coordinates: periodicity is off in all three
 * directions, so the pair scan never folds a separation into the cell.
 * Pairs come from the shared neighbour layer, and the per-species
 * coefficient tables are built once per species set.
 */
void ZBLPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto N = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  double *U = &out->energy;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  publishCauchyStress(out, strain, volume);
  const auto observe = [&](double fscale, double dx, double dy, double dz) {
    accumulatePairStrain(strain, fscale, dx, dy, dz);
  };
  *U = 0.0;
  std::fill(F, F + 3 * N, 0.0);
  if (N < 2) {
    return;
  }

  const auto tables = tablesFor(in.atmnrs, N);
  const std::size_t nTypes = tables->z.size();
  const ZblPairCoeffs *coeffs = tables->pairs.data();

  // Per-atom type index, so the pair loop indexes flat tables instead of
  // searching per pair.
  std::vector<int> typeIdx(static_cast<std::size_t>(N));
  for (long k = 0; k < N; ++k) {
    const auto it =
        std::lower_bound(tables->z.begin(), tables->z.end(), in.atmnrs[k]);
    typeIdx[static_cast<std::size_t>(k)] =
        static_cast<int>(std::distance(tables->z.begin(), it));
  }

  simd::PairSpec opt;
  opt.cutoff = cut_global;
  opt.skin = 0.0;
  opt.periodic = {{false, false, false}};

  static constexpr double free_box[9] = {1e6, 0.0, 0.0, 0.0, 1e6,
                                         0.0, 0.0, 0.0, 1e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  const double *box_use = box_ok ? in.box : free_box;

  simd::CachedPairs pairs;
  simd::fill_view(pairs, R, static_cast<std::size_t>(N), box_use, opt);
  const simd::PairView &view = pairs.view;
  const double rInner = cut_inner;
  const double rInnerSq = cut_inner_sq;
  const int *types = typeIdx.data();
  const double cutoff2 = cut_global * cut_global;
  double energy = 0.0;
  for (std::size_t i = 0; i < view.n; ++i) {
    double fxi = 0.0;
    double fyi = 0.0;
    double fzi = 0.0;
    const std::int32_t p0 = view.rows[i];
    const std::int32_t p1 = view.rows[i + 1];
    for (std::int32_t p = p0; p < p1; ++p) {
      const auto j = static_cast<std::size_t>(view.nbr[static_cast<std::size_t>(p)]);
      const double dx = R[3 * i] - R[3 * j];
      const double dy = R[3 * i + 1] - R[3 * j + 1];
      const double dz = R[3 * i + 2] - R[3 * j + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cutoff2) {
        continue;
      }
      const auto ti = static_cast<std::size_t>(types[i]);
      const auto tj = static_cast<std::size_t>(types[j]);
      const ZblPairCoeffs &coeff = coeffs[ti * nTypes + tj];
      const double r = std::sqrt(r2);
      double energy_pair = e_zbl(coeff, r) + coeff.sw5;
      double dEdr = dzbldr(coeff, r);
      if (r2 > rInnerSq) {
        const double t = r - rInner;
        energy_pair += t * t * t * (coeff.sw3 + coeff.sw4 * t);
        dEdr += t * t * (coeff.sw1 + coeff.sw2 * t);
      }
      const double fscale = -dEdr / r;
      energy += energy_pair;
      fxi += fscale * dx;
      fyi += fscale * dy;
      fzi += fscale * dz;
      F[3 * j] -= fscale * dx;
      F[3 * j + 1] -= fscale * dy;
      F[3 * j + 2] -= fscale * dz;
      observe(fscale, dx, dy, dz);
    }
    F[3 * i] += fxi;
    F[3 * i + 1] += fyi;
    F[3 * i + 2] += fzi;
  }
  *U = energy;
  publishCauchyStress(out, strain, volume);
}

} // namespace rgpot
