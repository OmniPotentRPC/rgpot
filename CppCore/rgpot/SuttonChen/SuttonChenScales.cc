// MIT License
// Copyright 2023--present rgpot developers

#include <cmath>
#include <stdexcept>

#include "rgpot/SuttonChen/SuttonChenScales.hpp"

namespace rgpot {
namespace {

constexpr int kFccLimit = 32;

SuttonChenConfig metal(double epsilon, double lattice) {
  SuttonChenConfig cfg;
  cfg.epsilon = epsilon;
  cfg.c = 34.408;
  cfg.a = lattice;
  cfg.n = 10;
  cfg.m = 8;
  return cfg;
}

} // namespace

double fcc_power_sum(int power) {
  double sum = 0.0;
  const double p = static_cast<double>(power);
  for (int i = -kFccLimit; i <= kFccLimit; ++i) {
    for (int j = -kFccLimit; j <= kFccLimit; ++j) {
      for (int k = -kFccLimit; k <= kFccLimit; ++k) {
        if (((i + j + k) & 1) != 0) {
          continue;
        }
        if (i == 0 && j == 0 && k == 0) {
          continue;
        }
        const double r2 = static_cast<double>(i * i + j * j + k * k);
        const double inv = std::sqrt(2.0 / r2);
        sum += std::pow(inv, p);
      }
    }
  }
  return sum;
}

double sutton_chen_fcc_energy(const SuttonChenConfig &cfg, double radius) {
  const double s_n = fcc_power_sum(cfg.n);
  const double s_m = fcc_power_sum(cfg.m);
  const double a_over_r = cfg.a / radius;
  const double repulsive =
      0.5 * std::pow(a_over_r, cfg.n) * s_n;
  const double density =
      std::pow(a_over_r, 0.5 * static_cast<double>(cfg.m)) * std::sqrt(s_m);
  return cfg.epsilon * (repulsive - cfg.c * density);
}

SuttonChenScales sutton_chen_scales(const SuttonChenConfig &cfg) {
  if (!(cfg.a > 0.0) || cfg.n < 1 || cfg.m < 1 ||
      static_cast<double>(cfg.n) <= 0.5 * static_cast<double>(cfg.m)) {
    throw std::invalid_argument("Sutton-Chen scales need a positive lattice "
                                "and n > m/2");
  }
  const double s_n = fcc_power_sum(cfg.n);
  const double s_m = fcc_power_sum(cfg.m);
  const double exponent =
      static_cast<double>(cfg.n) - 0.5 * static_cast<double>(cfg.m);
  const double rhs = (cfg.c * static_cast<double>(cfg.m) * std::sqrt(s_m)) /
                     (static_cast<double>(cfg.n) * s_n);
  SuttonChenScales out;
  out.nearest_neighbor = cfg.a / std::pow(rhs, 1.0 / exponent);
  out.well_depth = -sutton_chen_fcc_energy(cfg, out.nearest_neighbor);
  return out;
}

SuttonChenConfig sutton_chen_platinum() { return metal(1.98330e-2, 3.92); }

SuttonChenConfig sutton_chen_gold() { return metal(1.27930e-2, 4.08); }

} // namespace rgpot

namespace {

int export_scales(const rgpot_sc_config *cfg, rgpot_sc_scales *out) {
  if (cfg == nullptr || out == nullptr) {
    return 1;
  }
  try {
    const auto scales = rgpot::sutton_chen_scales(rgpot::SuttonChenConfig{
        .epsilon = cfg->epsilon,
        .c = cfg->c,
        .a = cfg->a,
        .n = cfg->n,
        .m = cfg->m,
        .cutoff = cfg->cutoff,
        .periodic = cfg->periodic != 0,
    });
    out->nearest_neighbor = scales.nearest_neighbor;
    out->well_depth = scales.well_depth;
  } catch (const std::exception &) {
    return 1;
  }
  return 0;
}

} // namespace

#if defined(_WIN32)
#define RGPOT_SC_API __declspec(dllexport)
#else
#define RGPOT_SC_API __attribute__((visibility("default")))
#endif

extern "C" RGPOT_SC_API int
rgpot_sutton_chen_scales(const rgpot_sc_config *cfg, rgpot_sc_scales *out) {
  return export_scales(cfg, out);
}
