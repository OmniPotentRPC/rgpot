// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Cleri-Rosato Gupta energy and analytic forces.
 *
 * Non-periodic, unbounded calls without stress or per-atom output delegate
 * directly to the C kernel @c rgpot_gupta. Periodic, truncated, stressed,
 * per-atom, or team-sized calls run on the shared neighbour machinery using the
 * same Cleri-Rosato pair expressions and embedding derivative.
 */

// clang-format off
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>
// clang-format on

#include "rgpot/Gupta/GuptaPot.hpp"
#include "rgpot/Gupta/gupta.h"
#include "rgpot/KeptPotential.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/stress.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define RGPOT_GUPTA_PAIR_INLINE __attribute__((always_inline))
#else
#define RGPOT_GUPTA_PAIR_INLINE
#endif

namespace {

/// Large cubic cell (Angstrom) used for open-boundary cluster evaluations so
/// minimum-image wrapping does not fold cluster pairs.
constexpr double kFreeBoxSideAngstrom = 1.0e6;

/// cutoff <= 0 keeps every pair; the unbounded search walks them all without
/// bins or a pair list.
double searchCutoff(const rgpot::GuptaConfig &cfg) {
  return cfg.cutoff > 0.0 ? cfg.cutoff
                          : std::numeric_limits<double>::infinity();
}

rgpot::GuptaConfig config_from_c(const rgpot_gupta_config &raw) {
  rgpot::GuptaConfig c;
  c.A = raw.A;
  c.xi = raw.xi;
  c.p = raw.p;
  c.q = raw.q;
  c.r0 = raw.r0;
  c.cutoff = raw.cutoff;
  c.periodic = raw.periodic != 0;
  return c;
}

} // namespace

namespace rgpot {

GuptaPot::GuptaPot(const GuptaConfig &c)
    : Potential(PotType::Gupta), m_config{c} {
  if (!(c.r0 > 0.0) || !std::isfinite(c.r0)) {
    throw std::invalid_argument("Gupta r0 must be positive and finite");
  }
  if (!std::isfinite(c.A) || !std::isfinite(c.xi) || !std::isfinite(c.p) ||
      !std::isfinite(c.q) || !std::isfinite(c.cutoff)) {
    throw std::invalid_argument("Gupta parameters must be finite");
  }
  if (c.periodic && !(c.cutoff > 0.0)) {
    throw std::invalid_argument("periodic Gupta needs a positive cutoff");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.A);
  fp.f64(c.xi);
  fp.f64(c.p);
  fp.f64(c.q);
  fp.f64(c.r0);
  fp.f64(c.cutoff);
  fp.u64(c.periodic ? 1 : 0);
  if (c.stress) {
    fp.u64(1);
  }
  m_paramsKey = fp.h;
}

void GuptaPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  compute(in, out, nullptr);
}

void GuptaPot::forcePerAtom(const ForceInput &in, ForceOut *out,
                            const PerAtomOut &atoms) const {
  clearPerAtom(atoms, in.nAtoms);
  compute(in, out, &atoms);
}

void GuptaPot::compute(const ForceInput &in, ForceOut *out,
                       const PerAtomOut *atoms) const {
  if (out == nullptr) {
    throw std::invalid_argument("GuptaPot: ForceOut is null");
  }
  const auto n_atoms = static_cast<long>(in.nAtoms);
  if (n_atoms > 0 && (in.pos == nullptr || out->F == nullptr)) {
    throw std::invalid_argument("GuptaPot: null positions or forces buffer");
  }
  const double *R = in.pos;
  double *F = out->F;
  out->energy = 0.0;
  out->variance = 0.0;
  for (long i = 0; i < n_atoms; ++i) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  const GuptaConfig cfg = m_config;
  double strain[6] = {};
  const double volume = cellVolume(in.box);
  if (n_atoms < 2) {
    if (cfg.stress) {
      publishCauchyStress(out, strain, volume);
    }
    return;
  }

  static constexpr double free_box[9] = {
      kFreeBoxSideAngstrom, 0.0, 0.0, 0.0, kFreeBoxSideAngstrom, 0.0, 0.0, 0.0,
      kFreeBoxSideAngstrom};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  if (cfg.periodic && !box_ok) {
    throw std::invalid_argument("periodic Gupta needs a real cell");
  }
  const auto n_size = static_cast<std::size_t>(n_atoms);

  // Free-cluster, unbounded calls without stress or per-atom output delegate
  // directly to the C kernel rgpot_gupta.
  if (!cfg.periodic && !(cfg.cutoff > 0.0) && (!cfg.stress || !(volume > 0.0)) &&
      atoms == nullptr && mode_ == nlist::Mode::Auto &&
      n_size < kTeamMinAtoms) {
    slot_.assign(n_size, 0);
    grad_.assign(3 * n_size, 0.0);
    rgpot_gupta(R, n_size, slot_.data(), 1, &cfg.A, &cfg.xi, &cfg.p, &cfg.q,
                &cfg.r0, &out->energy, grad_.data());
    for (std::size_t k = 0; k < 3 * n_size; ++k) {
      F[k] = -grad_[k];
    }
    if (cfg.stress) {
      publishCauchyStress(out, strain, volume);
    }
    return;
  }

  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
                   : std::array<bool, 3>{false, false, false};
  const double *box_use = box_ok ? in.box : free_box;

  const double A = cfg.A;
  const double xi2 = cfg.xi * cfg.xi;
  const double p = cfg.p;
  const double q = cfg.q;
  const double r0 = cfg.r0;
  double *atom_e = atoms != nullptr ? atoms->energy : nullptr;

  rho_.assign(n_size, 0.0);
  double *rho = rho_.data();
  double epair = 0.0;
  const double skin = mode_ == nlist::Mode::Stream ? 0.0 : cutoff / 5.0;
  nlist::updateCells(neighbours_, R, n_size, box_use, cutoff, skin,
                     nlist::Image::Round, mode_, periodic);
  const bool stream = neighbours_.streaming();

  const std::size_t from = teamThreshold(kTeamMinAtoms);
  const bool rows = n_size >= from;
  const int nt = nlist::teamSize(n_size, from, kTeamFrom);
  const bool by_row = rows && (nt > 1 || (stream && !neighbours_.allPairs()));
  std::vector<double> row_e(rows ? n_size : 0, 0.0);
  std::vector<double> row_s(rows && cfg.stress ? 6 * n_size : 0, 0.0);
  double *e_sum = rows ? row_e.data() : &epair;
  double *s_sum = !cfg.stress ? nullptr : (rows ? row_s.data() : strain);
  auto by_stride = [rows](auto walk) {
    if (rows) {
      walk(std::integral_constant<std::size_t, 1>{});
    } else {
      walk(std::integral_constant<std::size_t, 0>{});
    }
  };

  const auto lanes = static_cast<std::size_t>(by_row ? nt : 1);
  if (terms_.size() < lanes) {
    terms_.resize(lanes);
  }
  std::size_t expected = 0;
  if (!by_row) {
    expected = neighbours_.allPairs() ? n_size * (n_size - 1) / 2
               : stream               ? 0
                                      : neighbours_.slots() / 2;
  }
  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].setLimit(term_limit_ / lanes);
    terms_[t].record(t == 0 ? expected : 0);
  }

  auto pair_term = [=](double r2, double &phi, double &psi,
                       Terms &terms) RGPOT_GUPTA_PAIR_INLINE {
    double *kept = terms.push();
    const double r = std::sqrt(r2);
    const double u = r / r0 - 1.0;
    phi = 2.0 * A * std::exp(-p * u);
    psi = xi2 * std::exp(-2.0 * q * u);
    if (kept != nullptr) {
      kept[0] = r;
      kept[1] = -(p / r0) * phi;
      kept[2] = -(2.0 * q / r0) * psi;
    }
  };

  if (!by_row) {
    Terms &lane = terms_[0];
    by_stride([&](auto stride) {
      nlist::visitHalf(
          neighbours_, 0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double, double, double,
                     double r2) RGPOT_GUPTA_PAIR_INLINE {
            if (!(r2 > 0.0) || r2 > cutoff2) {
              return;
            }
            double phi = 0.0;
            double psi = 0.0;
            pair_term(r2, phi, psi, lane);
            rho[static_cast<std::size_t>(i)] += psi;
            rho[static_cast<std::size_t>(j)] += psi;
            e_sum[static_cast<std::size_t>(i) * decltype(stride)::value] += phi;
            if (atom_e != nullptr) {
              atom_e[static_cast<std::size_t>(i)] += 0.5 * phi;
              atom_e[static_cast<std::size_t>(j)] += 0.5 * phi;
            }
          });
    });
  } else {
    Terms *lane_terms = terms_.data();
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double, double, double,
            double r2) RGPOT_GUPTA_PAIR_INLINE {
          if (!(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          double phi = 0.0;
          double psi = 0.0;
          pair_term(r2, phi, psi, lane_terms[t]);
          rho[static_cast<std::size_t>(i)] += psi;
          if (j > i) {
            e_sum[static_cast<std::size_t>(i)] += phi;
          }
          if (atom_e != nullptr) {
            atom_e[static_cast<std::size_t>(i)] += 0.5 * phi;
          }
        });
  }

  weight_.assign(n_size, 0.0);
  double *weight = weight_.data();
  double total_e = rows ? 0.0 : epair;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho[i] > 0.0)) {
      continue;
    }
    const double root = std::sqrt(rho[i]);
    total_e -= root;
    weight[i] = 1.0 / root;
    if (atom_e != nullptr) {
      atom_e[i] -= root;
    }
  }

  for (std::size_t t = 0; t < lanes; ++t) {
    terms_[t].replay();
  }
  auto apply = [=](auto stride, int32_t i, int32_t j, double dx, double dy,
                   double dz, double r2, double *force, double *virial,
                   bool both, Terms &terms) RGPOT_GUPTA_PAIR_INLINE {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    double r = 0.0;
    double dphi = 0.0;
    double dpsi = 0.0;
    if (const double *kept = terms.next()) {
      r = kept[0];
      dphi = kept[1];
      dpsi = kept[2];
    } else {
      r = std::sqrt(r2);
      const double u = r / r0 - 1.0;
      const double phi = 2.0 * A * std::exp(-p * u);
      const double psi = xi2 * std::exp(-2.0 * q * u);
      dphi = -(p / r0) * phi;
      dpsi = -(2.0 * q / r0) * psi;
    }
    const double wi = weight[static_cast<std::size_t>(i)];
    const double wj = weight[static_cast<std::size_t>(j)];
    const double embed = -0.5 * (wi + wj);
    const double coef = (dphi + embed * dpsi) / r;
    const double fscale = -coef;
    const double fx = fscale * dx;
    const double fy = fscale * dy;
    const double fz = fscale * dz;
    force[3 * i] += fx;
    force[3 * i + 1] += fy;
    force[3 * i + 2] += fz;
    if (both) {
      force[3 * j] -= fx;
      force[3 * j + 1] -= fy;
      force[3 * j + 2] -= fz;
    }
    if (virial != nullptr && j > i) {
      accumulatePairStrain(
          virial +
              6 * static_cast<std::size_t>(i) * decltype(stride)::value,
          fscale, dx, dy, dz);
    }
    if (atoms != nullptr) {
      tallyAtomVirial(*atoms, static_cast<std::size_t>(i), 0.5, fscale, dx, dy,
                      dz);
      if (both) {
        tallyAtomVirial(*atoms, static_cast<std::size_t>(j), 0.5, fscale, dx,
                        dy, dz);
      }
    }
  };

  if (!by_row) {
    Terms &lane = terms_[0];
    double *virial = s_sum;
    by_stride([&](auto stride) {
      nlist::visitHalf(
          neighbours_, 0, n_atoms,
          [=, &lane](int32_t i, int32_t j, double dx, double dy, double dz,
                     double r2) RGPOT_GUPTA_PAIR_INLINE {
            apply(stride, i, j, dx, dy, dz, r2, F, virial, true, lane);
          });
    });
  } else {
    double *virial = s_sum;
    Terms *lane_terms = terms_.data();
    neighbours_.forRowTeam(
        nt,
        [=](int t, int32_t i, int32_t j, double dx, double dy, double dz,
            double r2) RGPOT_GUPTA_PAIR_INLINE {
          apply(std::integral_constant<std::size_t, 1>{}, i, j, dx, dy, dz, r2,
                F, virial, false, lane_terms[t]);
        });
  }

  if (rows) {
    for (std::size_t i = 0; i < n_size; ++i) {
      epair += row_e[i];
    }
    total_e += epair;
    if (cfg.stress) {
      for (std::size_t i = 0; i < n_size; ++i) {
        for (std::size_t k = 0; k < 6; ++k) {
          strain[k] += row_s[6 * i + k];
        }
      }
    }
  }

  out->energy = total_e;
  if (cfg.stress) {
    publishCauchyStress(out, strain, volume);
  }
}

void GuptaPot::densityStage(const HalfList &list, double *densities) const {
  const GuptaConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double xi2 = cfg.xi * cfg.xi;
  staged::forEachPair(
      list, [&](int i, int j, double, double, double, double r2, bool both) {
        if (!(r2 > 0.0) || r2 > cutoff2) {
          return;
        }
        const double r = std::sqrt(r2);
        const double u = r / cfg.r0 - 1.0;
        const double psi = xi2 * std::exp(-2.0 * cfg.q * u);
        densities[i] += psi;
        if (both) {
          densities[j] += psi;
        }
      });
}

void GuptaPot::forceStage(const HalfList &list, const double *densities,
                          double *forces, StageTally *tally) const {
  const GuptaConfig &cfg = m_config;
  const double cutoff = searchCutoff(cfg);
  const double cutoff2 = cutoff * cutoff;
  const double xi2 = cfg.xi * cfg.xi;
  std::vector<double> weight(list.nAtoms, 0.0);
  for (std::size_t i = 0; i < list.nAtoms; ++i) {
    if (densities[i] > 0.0) {
      weight[i] = 1.0 / std::sqrt(densities[i]);
    }
  }
  if (tally != nullptr) {
    for (std::size_t k = 0; k < list.nRows; ++k) {
      const int i = list.rows[k];
      if (densities[i] > 0.0) {
        staged::addSite(*tally, i, -std::sqrt(densities[i]));
      }
    }
  }
  staged::withTally(tally, [&](auto tallied) {
    staged::forEachPair(list, [&](int i, int j, double dx, double dy, double dz,
                                  double r2, bool both) {
      if (!(r2 > 0.0) || r2 > cutoff2) {
        return;
      }
      const double r = std::sqrt(r2);
      const double u = r / cfg.r0 - 1.0;
      const double phi = 2.0 * cfg.A * std::exp(-cfg.p * u);
      const double psi = xi2 * std::exp(-2.0 * cfg.q * u);
      const double dphi = -(cfg.p / cfg.r0) * phi;
      const double dpsi = -(2.0 * cfg.q / cfg.r0) * psi;
      const double wi = weight[static_cast<std::size_t>(i)];
      const double wj = weight[static_cast<std::size_t>(j)];
      const double embed = -0.5 * (wi + wj);
      const double fscale = -(dphi + embed * dpsi) / r;
      staged::addPair(tallied, forces, tally, i, j, both, phi, fscale, dx, dy,
                      dz);
    });
  });
}

} // namespace rgpot

namespace {

int export_force(const rgpot_gupta_config *cfg, int32_t natoms,
                 const double *positions, const double *box, double *energy,
                 double *forces) {
  if (cfg == nullptr || natoms < 0 || energy == nullptr ||
      (natoms > 0 && (positions == nullptr || forces == nullptr))) {
    return 1;
  }
  const auto n = static_cast<std::size_t>(natoms);
  rgpot::KeptKey key;
  key.f64(cfg->A)
      .f64(cfg->xi)
      .f64(cfg->p)
      .f64(cfg->q)
      .f64(cfg->r0)
      .f64(cfg->cutoff)
      .i64(cfg->periodic != 0 ? 1 : 0)
      .atoms(n, nullptr);
  try {
    *energy = rgpot::Kept<rgpot::GuptaPot>::with(
        key,
        [&] { return std::make_unique<rgpot::GuptaPot>(config_from_c(*cfg)); },
        [&](const rgpot::GuptaPot &pot) {
          rgpot::ForceOut out{};
          out.F = forces;
          out.energy = 0.0;
          out.variance = 0.0;
          out.has_stress = 0;
          const rgpot::ForceInput in{n, positions, nullptr, box};
          pot.forceImpl(in, &out);
          return out.energy;
        });
  } catch (const std::exception &) {
    return 1;
  }
  return 0;
}

} // namespace

extern "C" RGPOT_GUPTA_API int
rgpot_gupta_energy_forces(const rgpot_gupta_config *cfg, int32_t natoms,
                          const double *positions, const double *box,
                          double *energy, double *forces) {
  return export_force(cfg, natoms, positions, box, energy, forces);
}
