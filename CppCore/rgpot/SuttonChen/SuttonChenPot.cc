// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Sutton-Chen energy and analytic forces.
 *
 * Each unordered pair contributes epsilon * (a/r)^n once, which is the
 * 1/2 sum_i sum_{j != i} term. The density sum has no 1/2. The force
 * on atom i along d = r_i - r_j is fscale * d, with
 * fscale = -(dE/dr) / r.
 */

// clang-format off
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
// clang-format on

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/PairListCache.hpp"

namespace {

double ipow(double base, int exp) {
  double result = 1.0;
  while (exp > 0) {
    if ((exp & 1) != 0) {
      result *= base;
    }
    base *= base;
    exp >>= 1;
  }
  return result;
}

/// (a/r)^exp from ar2 = (a/r)^2.
double pow_ar(double ar2, int exp) {
  double p = ipow(ar2, exp / 2);
  if ((exp % 2) != 0) {
    p *= std::sqrt(ar2);
  }
  return p;
}

rgpot::SuttonChenConfig config_from_c(const rgpot_sc_config &raw) {
  rgpot::SuttonChenConfig c;
  c.epsilon = raw.epsilon;
  c.c = raw.c;
  c.a = raw.a;
  c.n = raw.n;
  c.m = raw.m;
  c.cutoff = raw.cutoff;
  c.periodic = raw.periodic != 0;
  return c;
}

} // namespace

namespace rgpot {

SuttonChenPot::SuttonChenPot(const SuttonChenConfig &c)
    : Potential(PotType::SuttonChen), m_config{c} {
  if (c.n < 1 || c.m < 1) {
    throw std::invalid_argument("Sutton-Chen exponents n and m must be >= 1");
  }
  if (!(c.a > 0.0) || !std::isfinite(c.epsilon) || !std::isfinite(c.c) ||
      !std::isfinite(c.cutoff)) {
    throw std::invalid_argument("Sutton-Chen parameters are not finite");
  }
  if (c.periodic && !(c.cutoff > 0.0)) {
    throw std::invalid_argument(
        "periodic Sutton-Chen needs a positive cutoff");
  }
  Fnv1a fp;
  fp.u64(kKernelVersion);
  fp.f64(c.epsilon);
  fp.f64(c.c);
  fp.f64(c.a);
  fp.u64(static_cast<uint64_t>(c.n));
  fp.u64(static_cast<uint64_t>(c.m));
  fp.f64(c.cutoff);
  fp.u64(c.periodic ? 1 : 0);
  m_paramsKey = fp.h;
}

void SuttonChenPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  const auto n_atoms = static_cast<long>(in.nAtoms);
  const double *R = in.pos;
  double *F = out->F;
  out->energy = 0.0;
  out->variance = 0.0;
  for (long i = 0; i < n_atoms; ++i) {
    F[3 * i] = 0.0;
    F[3 * i + 1] = 0.0;
    F[3 * i + 2] = 0.0;
  }
  if (n_atoms < 2) {
    return;
  }

  const SuttonChenConfig cfg = m_config;
  nlist::CachedPairList::Options opt;
  opt.cutoff = (cfg.cutoff > 0.0) ? cfg.cutoff : 1.0e6;
  opt.periodic = cfg.periodic ? std::array<bool, 3>{true, true, true}
                              : std::array<bool, 3>{false, false, false};

  static constexpr double free_box[9] = {1.0e6, 0.0, 0.0, 0.0, 1.0e6,
                                         0.0,   0.0, 0.0, 1.0e6};
  const bool box_ok = in.box != nullptr && in.box[0] > 0.0 && in.box[4] > 0.0 &&
                      in.box[8] > 0.0;
  if (cfg.periodic && !box_ok) {
    throw std::invalid_argument("periodic Sutton-Chen needs a real cell");
  }
  const double *box_use = box_ok ? in.box : free_box;

  const double epsilon = cfg.epsilon;
  const double c_coef = cfg.c;
  const double lattice = cfg.a;
  const int n_exp = cfg.n;
  const int m_exp = cfg.m;
  const auto n_size = static_cast<std::size_t>(n_atoms);

  std::vector<double> rho(n_size, 0.0);
  double epair = 0.0;
  auto &pool = nlist::PairListCache::global();
  pool.evaluate(
      R, n_size, box_use, opt,
      [&](int32_t i, int32_t j, double, double, double, double r2) {
        if (!(r2 > 0.0)) {
          return;
        }
        const double ar2 = (lattice * lattice) / r2;
        const double phi = pow_ar(ar2, m_exp);
        const double rep = pow_ar(ar2, n_exp);
        rho[static_cast<std::size_t>(i)] += phi;
        rho[static_cast<std::size_t>(j)] += phi;
        epair += epsilon * rep;
      });

  std::vector<double> weight(n_size, 0.0);
  double eemb = 0.0;
  for (std::size_t i = 0; i < n_size; ++i) {
    if (!(rho[i] > 0.0)) {
      continue;
    }
    const double root = std::sqrt(rho[i]);
    eemb -= epsilon * c_coef * root;
    weight[i] = epsilon * c_coef / (2.0 * root);
  }

  pool.evaluate(
      R, n_size, box_use, opt,
      [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
        if (!(r2 > 0.0)) {
          return;
        }
        const double inv_r2 = 1.0 / r2;
        const double ar2 = (lattice * lattice) * inv_r2;
        const double phi = pow_ar(ar2, m_exp);
        const double rep = pow_ar(ar2, n_exp);
        const double wi = weight[static_cast<std::size_t>(i)];
        const double wj = weight[static_cast<std::size_t>(j)];
        const double fscale =
            (epsilon * static_cast<double>(n_exp) * rep -
             static_cast<double>(m_exp) * phi * (wi + wj)) *
            inv_r2;
        const double fx = fscale * dx;
        const double fy = fscale * dy;
        const double fz = fscale * dz;
        F[3 * i] += fx;
        F[3 * i + 1] += fy;
        F[3 * i + 2] += fz;
        F[3 * j] -= fx;
        F[3 * j + 1] -= fy;
        F[3 * j + 2] -= fz;
      });

  out->energy = epair + eemb;
}

} // namespace rgpot

namespace {

int export_force(const rgpot_sc_config *cfg, int32_t natoms,
                 const double *positions, const double *box, double *energy,
                 double *forces) {
  if (cfg == nullptr || natoms < 0 || energy == nullptr ||
      (natoms > 0 && (positions == nullptr || forces == nullptr))) {
    return 1;
  }
  try {
    const rgpot::SuttonChenPot pot(config_from_c(*cfg));
    rgpot::ForceOut out{};
    out.F = forces;
    out.energy = 0.0;
    out.variance = 0.0;
    out.has_stress = 0;
    const rgpot::ForceInput in{static_cast<std::size_t>(natoms), positions,
                               nullptr, box};
    pot.forceImpl(in, &out);
    *energy = out.energy;
  } catch (const std::exception &) {
    return 1;
  }
  return 0;
}

} // namespace

extern "C" RGPOT_SC_API int
rgpot_sutton_chen_energy_forces(const rgpot_sc_config *cfg, int32_t natoms,
                        const double *positions, const double *box,
                        double *energy, double *forces) {
  return export_force(cfg, natoms, positions, box, energy, forces);
}
