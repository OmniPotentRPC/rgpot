// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D2Pot/D2CpmdPot.hpp"

#include "rgpot/D2Pot/D2Cell.hpp"
#include "rgpot/ParamHash.hpp"
#include "rgpot/units.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

namespace {

// OpenCPMD 062582b, src/vdwin_utils.mod.F90 (empvdwin): vdwc6 in
// J nm^6 mol^-1 and vdwr0 in Angstrom, H to Rn. These are Grimme 2006,
// Table 1, as CPMD types them, B's C6 of 3.31 included; elements 87 to 99
// have no parameters and give nothing.
constexpr std::array<double, 86> kCpmdC6 = {
    0.14, 0.08, 1.61, 1.61, 3.31, 1.75, 1.23, 0.70,
    0.75, 0.63, 5.71, 5.71, 10.79, 9.23, 7.84, 5.57,
    5.07, 4.61, 10.80, 10.80, 10.80, 10.80, 10.80, 10.80,
    10.80, 10.80, 10.80, 10.80, 10.80, 10.80, 16.99, 17.10,
    16.37, 12.64, 12.47, 12.01, 24.67, 24.67, 24.67, 24.67,
    24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67, 24.67,
    37.32, 38.71, 38.44, 31.74, 31.50, 29.99, 315.275, 226.994,
    176.252, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68,
    140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 140.68, 105.112,
    81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 81.24, 57.364,
    57.254, 63.162, 63.540, 55.283, 57.171, 56.64,
};
constexpr std::array<double, 86> kCpmdR0 = {
    1.001, 1.012, 0.825, 1.408, 1.485, 1.452, 1.397, 1.342,
    1.287, 1.243, 1.144, 1.364, 1.639, 1.716, 1.705, 1.683,
    1.639, 1.595, 1.485, 1.474, 1.562, 1.562, 1.562, 1.562,
    1.562, 1.562, 1.562, 1.562, 1.562, 1.562, 1.650, 1.727,
    1.760, 1.771, 1.749, 1.727, 1.628, 1.606, 1.639, 1.639,
    1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639, 1.639,
    1.672, 1.804, 1.881, 1.892, 1.892, 1.881, 1.802, 1.762,
    1.720, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753,
    1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.753, 1.788,
    1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.772, 1.758,
    1.986, 1.944, 1.898, 2.005, 1.991, 1.924,
};

// convfac in empvdwin and fbohr in src/cnst.mod.F90.
constexpr double kCpmdC6Conv = 1.733979e1;
constexpr double kCpmdBohr = 0.529177210859;
constexpr double kCpmdFbohr = 1.0 / kCpmdBohr;
// reps in vdw: a pair closer than this many bohr contributes nothing.
constexpr double kCpmdMinPair = 1.0e-2;
constexpr int kCpmdMaxZ = 99;
constexpr double kCellTol = 1.0e-12;

std::string upper(std::string_view s) {
  std::string out(s);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return out;
}

/// CPMD's pbc3 for an orthorhombic cell (labels 5 and 15): per axis,
/// x - L * dint(dmod(x, L) / (L / 2)) applied to dmod(x, L).
double wrap_axis(double x, double length) {
  const double y = std::fmod(x, length);
  return y - length * std::trunc(y / (0.5 * length));
}

double dot3(const std::array<double, 3> &a, const std::array<double, 3> &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm3(const std::array<double, 3> &v) { return std::sqrt(dot3(v, v)); }

bool nearly(double a, double b) { return std::abs(a - b) <= kCellTol; }

std::array<double, 3> row(const std::array<double, 9> &h, int i) {
  return {h[3 * i], h[3 * i + 1], h[3 * i + 2]};
}

bool is_fcc_cell(const std::array<double, 9> &h) {
  const auto a = row(h, 0);
  const auto b = row(h, 1);
  const auto c = row(h, 2);
  const double la = norm3(a);
  const double lb = norm3(b);
  const double lc = norm3(c);
  if (!(nearly(la, lb) && nearly(lb, lc))) {
    return false;
  }
  const double c12 = dot3(a, b) / (la * lb);
  const double c13 = dot3(a, c) / (la * lc);
  const double c23 = dot3(b, c) / (lb * lc);
  return nearly(c12, 0.5) && nearly(c13, 0.5) && nearly(c23, 0.5);
}

bool is_trigonal_cell(const std::array<double, 9> &h) {
  const auto a = row(h, 0);
  const auto b = row(h, 1);
  const auto c = row(h, 2);
  const double la = norm3(a);
  const double lb = norm3(b);
  const double lc = norm3(c);
  if (!(nearly(la, lb) && nearly(lb, lc))) {
    return false;
  }
  const double c12 = dot3(a, b) / (la * lb);
  const double c13 = dot3(a, c) / (la * lc);
  const double c23 = dot3(b, c) / (lb * lc);
  return nearly(c12, c13) && nearly(c12, c23) && !nearly(c12, 0.0) &&
         !nearly(c12, 0.5);
}

bool is_hexagonal_cell(const std::array<double, 9> &h) {
  const auto a = row(h, 0);
  const auto b = row(h, 1);
  const auto c = row(h, 2);
  const double la = norm3(a);
  const double lb = norm3(b);
  const double lc = norm3(c);
  if (!(nearly(la, lb) && !nearly(lb, lc))) {
    return false;
  }
  const double c12 = dot3(a, b) / (la * lb);
  const double c13 = dot3(a, c) / (la * lc);
  const double c23 = dot3(b, c) / (lb * lc);
  return nearly(c12, -0.5) && nearly(c13, 0.0) && nearly(c23, 0.0);
}

bool is_monoclinic_cell(const std::array<double, 9> &h) {
  const auto a = row(h, 0);
  const auto b = row(h, 1);
  const auto c = row(h, 2);
  const double la = norm3(a);
  const double lb = norm3(b);
  const double lc = norm3(c);
  const double c12 = dot3(a, b) / (la * lb);
  const double c13 = dot3(a, c) / (la * lc);
  const double c23 = dot3(b, c) / (lb * lc);
  const int near_90 = static_cast<int>(nearly(c12, 0.0)) +
                      static_cast<int>(nearly(c13, 0.0)) +
                      static_cast<int>(nearly(c23, 0.0));
  return near_90 == 2;
}

bool uses_symmetry_specific_pbc3(const std::array<double, 9> &h) {
  // OpenCPMD 062582b, src/vdwin_utils.mod.F90, routine `pbc3`: FCC,
  // trigonal, hexagonal and monoclinic lattice kinds each have their own
  // branch. These branches are not equivalent to rounding each fractional
  // component independently, so we reproduce them with an image search over
  // the wrapped vector.
  return is_fcc_cell(h) || is_trigonal_cell(h) || is_hexagonal_cell(h) ||
         is_monoclinic_cell(h);
}

void wrap_symmetry_specific(const d2::detail::Cell &cell, double *rlp) {
  double s[3];
  cell.fractional(rlp, s);
  std::array<double, 3> best{rlp[0], rlp[1], rlp[2]};
  double best2 = best[0] * best[0] + best[1] * best[1] + best[2] * best[2];
  for (int nx = -1; nx <= 1; ++nx) {
    for (int ny = -1; ny <= 1; ++ny) {
      for (int nz = -1; nz <= 1; ++nz) {
        const double shifted[3] = {s[0] - static_cast<double>(nx),
                                   s[1] - static_cast<double>(ny),
                                   s[2] - static_cast<double>(nz)};
        double cand[3];
        cell.cartesian(shifted, cand);
        const double r2 = cand[0] * cand[0] + cand[1] * cand[1] + cand[2] * cand[2];
        if (r2 + kCellTol < best2) {
          best2 = r2;
          best = {cand[0], cand[1], cand[2]};
        }
      }
    }
  }
  rlp[0] = best[0];
  rlp[1] = best[1];
  rlp[2] = best[2];
}

} // namespace

D2CpmdConfig d2_cpmd_config_for(std::string_view functional) {
  // set_s6 in src/vdwin_utils.mod.F90.
  struct Row {
    const char *name;
    double s6;
  };
  static constexpr Row kRows[] = {
      {"BLYP", 1.20}, {"B3LYP", 1.05}, {"BP86", 1.05}, {"TPSS", 1.00},
      {"PBE", 0.75},  {"REVPBE", 1.25}, {"PBE0", 0.60},
  };
  const std::string key = upper(functional);
  for (const Row &r : kRows) {
    if (key == r.name) {
      D2CpmdConfig c;
      c.s6 = r.s6;
      return c;
    }
  }
  throw std::invalid_argument("CPMD's D2 has no s6 for functional '" + std::string(functional) +
                              "'");
}

D2CpmdPot::D2CpmdPot(D2CpmdConfig cfg) : Potential(PotType::D2Cpmd), cfg_(cfg) {
  if (!std::isfinite(cfg_.s6) || !std::isfinite(cfg_.alpha) || !(cfg_.vdweps > 0.0) ||
      !std::isfinite(cfg_.vdweps)) {
    throw std::invalid_argument("CPMD's D2 needs finite s6 and alpha and a positive vdweps");
  }
}

void D2CpmdPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  using units::BOHR_TO_ANGSTROM;
  using units::HARTREE_TO_EV;
  const std::size_t n = in.nAtoms;
  if (n > 0 && in.atmnrs == nullptr) {
    throw std::invalid_argument("CPMD's D2 needs atomic numbers");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (in.atmnrs[i] < 1 || in.atmnrs[i] > kCpmdMaxZ) {
      throw std::invalid_argument("CPMD's D2 takes atomic numbers 1 to 99, got " +
                                  std::to_string(in.atmnrs[i]));
    }
  }
  std::vector<double> x(3 * n);
  for (std::size_t k = 0; k < 3 * n; ++k) {
    x[k] = in.pos[k] * kCpmdFbohr;
  }
  const d2::detail::Cell cell = d2::detail::Cell::from_box(in.box, kCpmdBohr);
  const auto &h = cell.h;
  const bool orthorhombic = cell.periodic && h[1] == 0.0 && h[2] == 0.0 && h[3] == 0.0 &&
                            h[5] == 0.0 && h[6] == 0.0 && h[7] == 0.0;
  const bool symmetry_pbc3 = cell.periodic && uses_symmetry_specific_pbc3(h);
  const double rpow_max = 1.0 / cfg_.vdweps;

  std::vector<double> f(3 * n, 0.0);
  double evdw = 0.0;
  std::array<double, 9> w{};
  for (std::size_t l = 0; l < n; ++l) {
    for (std::size_t m = 0; m < l; ++m) {
      const int zl = in.atmnrs[l];
      const int zm = in.atmnrs[m];
      constexpr int kTabulated = static_cast<int>(kCpmdC6.size());
      if (zl > kTabulated || zm > kTabulated) {
        continue;
      }
      double rlp[3] = {x[3 * l] - x[3 * m], x[3 * l + 1] - x[3 * m + 1],
                       x[3 * l + 2] - x[3 * m + 2]};
      if (orthorhombic) {
        for (int a = 0; a < 3; ++a) {
          rlp[a] = wrap_axis(rlp[a], h[4 * a]);
        }
      } else if (symmetry_pbc3) {
        wrap_symmetry_specific(cell, rlp);
      } else if (cell.periodic) {
        double s[3];
        cell.fractional(rlp, s);
        for (double &v : s) {
          v -= std::round(v);
        }
        cell.cartesian(s, rlp);
      }
      const double rlm2 = rlp[0] * rlp[0] + rlp[1] * rlp[1] + rlp[2] * rlp[2];
      const double rlm1 = std::sqrt(rlm2);
      if (rlm1 < kCpmdMinPair) {
        continue;
      }
      const double vdwst = std::sqrt(kCpmdC6[zl - 1] * kCpmdC6[zm - 1]) * kCpmdC6Conv;
      const double vdwrm = (kCpmdR0[zl - 1] + kCpmdR0[zm - 1]) * kCpmdFbohr;
      const double rlm6 = rlm2 * rlm2 * rlm2;
      const double rpow = cfg_.alpha * (rlm1 / vdwrm - 1.0);
      if (!(std::abs(rpow) < rpow_max)) {
        continue;
      }
      const double rexp = 1.0 + std::exp(-rpow);
      const double rphi = -cfg_.s6 * vdwst / (rlm6 * rexp);
      const double ffac = cfg_.s6 * vdwst / (rlm6 * rexp * rlm1) *
                          (6.0 / rlm1 - cfg_.alpha * (rexp - 1.0) / (vdwrm * rexp));
      evdw += rphi;
      for (int a = 0; a < 3; ++a) {
        f[3 * l + a] -= ffac * rlp[a];
        f[3 * m + a] += ffac * rlp[a];
        for (int b = 0; b < 3; ++b) {
          w[3 * a + b] += ffac * rlp[a] * rlp[b];
        }
      }
    }
  }

  out->energy = evdw * HARTREE_TO_EV;
  const double fconv = HARTREE_TO_EV / BOHR_TO_ANGSTROM;
  for (std::size_t k = 0; k < 3 * n; ++k) {
    out->F[k] = f[k] * fconv;
  }
  if (cell.periodic) {
    for (int k = 0; k < 9; ++k) {
      out->stress[k] = w[k] / cell.volume * units::HARTREE_PER_BOHR3_TO_EV_PER_ANGSTROM3;
    }
    out->has_stress = 1;
  }
}

uint64_t D2CpmdPot::paramsKey() const noexcept {
  Fnv1a h;
  h.str("rgpot::D2CpmdPot");
  h.f64(cfg_.s6);
  h.f64(cfg_.alpha);
  h.f64(cfg_.vdweps);
  return h.h;
}

} // namespace rgpot
