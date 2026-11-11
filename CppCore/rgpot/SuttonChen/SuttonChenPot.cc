// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Sutton-Chen energy and analytic forces.
 *
 * Each unordered pair contributes epsilon * (a/r)^n once, which is the
 * 1/2 sum_i sum_{j != i} term. The density sum has no 1/2. The force
 * on atom i along d = r_i - r_j is fscale * d, with
 * fscale = -(dE/dr) / r.
 *
 * The energy depends on positions only through pair distances, so with
 * SuttonChenConfig::stress the force pass also sums -fscale * d (x) d,
 * which is dE/dstrain for the embedding and pair terms together.
 */

// clang-format off
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
// clang-format on

#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/nlist/Neighbours.hpp"
#include "rgpot/stress.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

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
  if (c.stress) {
    // Mixed in only when set, so the default key keeps its old value.
    fp.u64(1);
  }
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
  const SuttonChenConfig cfg = m_config;
  double strain[6] = {};
  if (n_atoms < 2) {
    if (cfg.stress) {
      publishCauchyStress(out, strain, cellVolume(in.box));
    }
    return;
  }

  const double cutoff = (cfg.cutoff > 0.0) ? cfg.cutoff : 1.0e6;
  const double cutoff2 = cutoff * cutoff;
  const std::array<bool, 3> periodic =
      cfg.periodic ? std::array<bool, 3>{true, true, true}
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
  nlist::Spec spec;
  spec.cutoff = cutoff;
  spec.skin = cutoff / 5.0;
  spec.periodic = periodic;
  spec.image = nlist::Image::Round;
  spec.mode = mode_;
  neighbours_.update(R, n_size, box_use, spec);
  const int nt = nlist::threadsRequested();
  const bool stream = neighbours_.streaming();
  // Accepted pairs from the one cell walk. The force pass reads these
  // instead of walking the cells again.
  std::vector<std::int32_t> hit_i;
  std::vector<std::int32_t> hit_j;
  std::vector<std::vector<std::int32_t>> part_i;
  std::vector<std::vector<std::int32_t>> part_j;
  if (stream && nt <= 1) {
    hit_i.reserve(n_size * 8);
    hit_j.reserve(n_size * 8);
  }
  if (stream && nt > 1) {
    part_i.resize(static_cast<std::size_t>(nt));
    part_j.resize(static_cast<std::size_t>(nt));
  }
  auto pair_term = [&](double r2, double &phi, double &rep) {
    const double ar2 = (lattice * lattice) / r2;
    phi = pow_ar(ar2, m_exp);
    rep = pow_ar(ar2, n_exp);
  };
  if (nt == 1) {
    neighbours_.forDirected(
        0, n_atoms, [&](int32_t i, int32_t j, double, double, double, double r2) {
          if (!(j > i) || !(r2 > 0.0) || r2 > cutoff2) {
            return;
          }
          double phi = 0.0;
          double rep = 0.0;
          pair_term(r2, phi, rep);
          rho[static_cast<std::size_t>(i)] += phi;
          rho[static_cast<std::size_t>(j)] += phi;
          epair += epsilon * rep;
          if (stream) {
            hit_i.push_back(i);
            hit_j.push_back(j);
          }
        });
  } else {
    std::vector<nlist::Range> parts(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      parts[static_cast<std::size_t>(t)] = nlist::partition(n_atoms, t, nt);
    }
    std::vector<double> partial(static_cast<std::size_t>(nt), 0.0);
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int t = omp_get_thread_num();
#else
      const int t = 0;
#endif
      const auto range = parts[static_cast<std::size_t>(t)];
      double local = 0.0;
      neighbours_.forDirected(
          range.begin, range.end,
          [&](int32_t i, int32_t j, double, double, double, double r2) {
            if (!(r2 > 0.0) || r2 > cutoff2) {
              return;
            }
            double phi = 0.0;
            double rep = 0.0;
            pair_term(r2, phi, rep);
            rho[static_cast<std::size_t>(i)] += phi;
            if (j > i) {
              local += epsilon * rep;
            }
            if (stream) {
              part_i[static_cast<std::size_t>(t)].push_back(i);
              part_j[static_cast<std::size_t>(t)].push_back(j);
            }
          });
      partial[static_cast<std::size_t>(t)] = local;
    }
    for (double part : partial) {
      epair += part;
    }
  }

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

  auto apply = [&](int32_t i, int32_t j, double dx, double dy, double dz,
                   double r2, double *force, double *virial, bool both) {
    if (!(r2 > 0.0) || r2 > cutoff2) {
      return;
    }
    const double inv_r2 = 1.0 / r2;
    const double ar2 = (lattice * lattice) * inv_r2;
    const double phi = pow_ar(ar2, m_exp);
    const double rep = pow_ar(ar2, n_exp);
    const double wi = weight[static_cast<std::size_t>(i)];
    const double wj = weight[static_cast<std::size_t>(j)];
    const double fscale = (epsilon * static_cast<double>(n_exp) * rep -
                           static_cast<double>(m_exp) * phi * (wi + wj)) *
                          inv_r2;
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
      accumulatePairStrain(virial, fscale, dx, dy, dz);
    }
  };
  auto round_visit = [](double t) {
    if (!std::isfinite(t)) {
      return 0.0;
    }
    const double shifted = t + std::copysign(0.5, t);
    if (!std::isfinite(shifted) || shifted >= 9223372036854775808.0 ||
        shifted <= -9223372036854775808.0) {
      return 0.0;
    }
    return static_cast<double>(static_cast<long long>(shifted));
  };
  const double len0 = box_use[0];
  const double len1 = box_use[4];
  const double len2 = box_use[8];
  const double inv0 = periodic[0] && len0 != 0.0 ? 1.0 / len0 : 0.0;
  const double inv1 = periodic[1] && len1 != 0.0 ? 1.0 / len1 : 0.0;
  const double inv2 = periodic[2] && len2 != 0.0 ? 1.0 / len2 : 0.0;
  auto separation = [&](std::int32_t i, std::int32_t j, double &dx, double &dy,
                        double &dz, double &r2) {
    const auto ui = static_cast<std::size_t>(i) * 3;
    const auto uj = static_cast<std::size_t>(j) * 3;
    dx = R[ui] - R[uj];
    dy = R[ui + 1] - R[uj + 1];
    dz = R[ui + 2] - R[uj + 2];
    if (inv0 != 0.0) {
      dx -= len0 * round_visit(dx * inv0);
    }
    if (inv1 != 0.0) {
      dy -= len1 * round_visit(dy * inv1);
    }
    if (inv2 != 0.0) {
      dz -= len2 * round_visit(dz * inv2);
    }
    r2 = dx * dx + dy * dy + dz * dz;
  };
  if (nt == 1) {
    if (stream) {
      for (std::size_t p = 0; p < hit_i.size(); ++p) {
        double dx = 0.0;
        double dy = 0.0;
        double dz = 0.0;
        double r2 = 0.0;
        separation(hit_i[p], hit_j[p], dx, dy, dz, r2);
        apply(hit_i[p], hit_j[p], dx, dy, dz, r2, F,
              cfg.stress ? strain : nullptr, true);
      }
    } else {
      neighbours_.forDirected(
          0, n_atoms,
          [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
            if (!(j > i)) {
              return;
            }
            apply(i, j, dx, dy, dz, r2, F, cfg.stress ? strain : nullptr, true);
          });
    }
  } else {
    std::vector<nlist::Range> parts(static_cast<std::size_t>(nt));
    for (int t = 0; t < nt; ++t) {
      parts[static_cast<std::size_t>(t)] = nlist::partition(n_atoms, t, nt);
    }
    std::vector<double> owned(static_cast<std::size_t>(nt) * n_size * 3, 0.0);
    std::vector<double> vir(static_cast<std::size_t>(nt) * 6, 0.0);
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int t = omp_get_thread_num();
#else
      const int t = 0;
#endif
      const auto range = parts[static_cast<std::size_t>(t)];
      double *force = owned.data() + static_cast<std::size_t>(t) * n_size * 3;
      double *virial = cfg.stress ? vir.data() + static_cast<std::size_t>(t) * 6
                                 : nullptr;
      if (stream) {
        const auto &pi = part_i[static_cast<std::size_t>(t)];
        const auto &pj = part_j[static_cast<std::size_t>(t)];
        for (std::size_t p = 0; p < pi.size(); ++p) {
          double dx = 0.0;
          double dy = 0.0;
          double dz = 0.0;
          double r2 = 0.0;
          separation(pi[p], pj[p], dx, dy, dz, r2);
          apply(pi[p], pj[p], dx, dy, dz, r2, force, virial, false);
        }
      } else {
        neighbours_.forDirected(
            range.begin, range.end,
            [&](int32_t i, int32_t j, double dx, double dy, double dz, double r2) {
              apply(i, j, dx, dy, dz, r2, force, virial, false);
            });
      }
    }
    for (int t = 0; t < nt; ++t) {
      const double *force = owned.data() + static_cast<std::size_t>(t) * n_size * 3;
      for (std::size_t k = 0; k < n_size * 3; ++k) {
        F[k] += force[k];
      }
      if (cfg.stress) {
        for (int k = 0; k < 6; ++k) {
          strain[k] += vir[static_cast<std::size_t>(t) * 6 + static_cast<std::size_t>(k)];
        }
      }
    }
  }

  out->energy = epair + eemb;
  if (cfg.stress) {
    publishCauchyStress(out, strain, cellVolume(in.box));
  }
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
