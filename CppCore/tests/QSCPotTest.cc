// MIT License
// Copyright 2023--present rgpot developers
//
// Quantum Sutton-Chen (QSCPot) checks:
// - Published Q-SC parameters and cohesive energies from Qi, Cagin, Kimura
//   and Goddard, Phys. Rev. B 59, 3527 (1999), doi:10.1103/PhysRevB.59.3527,
//   and Ozdemir Kart et al., J. Non-Cryst. Solids 337, 101 (2004),
//   doi:10.1016/j.jnoncrysol.2004.03.121.
// - Combination rules of Eq. (2) of Qi et al. (after Rafii-Tabar and Sutton,
//   Philos. Mag. Lett. 63, 217 (1991), doi:10.1080/09500839108205994),
//   including half-integer exponents on Cu-Ag; the density uses the mixed
//   a_ij and m_ij of Eq. (1c).
// - Doye and Wales 10-8 cluster minima (Pt38 Oh and Pt75 D5h) via
//   setElementParameter, and bit-parity with SuttonChenPot.
// - Analytic forces, Cauchy stress, per-atom split, staged evaluation and
//   analytic Hessian-vector products against finite-difference oracles.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "lsan_libomp.hpp"
#include "rgpot/HessianVector.hpp"
#include "rgpot/QSC/QSCPot.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using rgpot::types::AtomMatrix;

namespace {

constexpr int kVoigt[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
constexpr double kDimerTol = 1.0e-14;
constexpr double kZeroForceTol = 1.0e-15;
constexpr double kShellPad = 1.0e-9;
constexpr double kExactEcohTol = 1.0e-11;
constexpr double kPublishedEcohTol = 5.0e-3;
constexpr double kOracleTol = 1.0e-12;
constexpr double kDoyeClusterTol = 5.0e-5;
constexpr double kFdForceStep = 1.0e-4;
constexpr double kFdForceTol = 1.0e-8;
constexpr double kFdStrainStep = 1.0e-6;
constexpr double kFdStressTol = 1.0e-6;
constexpr double kHvpFloor = 1.0e-3;
constexpr double kHvpRelTol = 1.0e-9;

std::array<std::array<double, 3>, 3> cube(double length) {
  return {{{length, 0.0, 0.0}, {0.0, length, 0.0}, {0.0, 0.0, length}}};
}

struct QscRow {
  int z;
  const char *symbol;
  double n;
  double m;
  double epsilon;
  double c;
  double a;
  double published_ecoh;
  double exact_2a_ecoh;
};

// Ni and Cu rows: Table I of Qi, Cagin, Kimura and Goddard, Phys. Rev. B 59,
// 3527 (1999), doi:10.1103/PhysRevB.59.3527 (Cu a = 3.6030; eOn lists 3.5200).
// Other rows: the eight-metal Q-SC table of Ozdemir Kart et al., J. Non-Cryst.
// Solids 337, 101 (2004), Table 1, as in eOn client/potentials/QSC/Parameters.h.
// Table I prints a different Ag row (4.0072 meV, c = 94.948) that is not used.
constexpr std::array<QscRow, 8> kQscTable = {{
    {28, "Ni", 10.0, 5.0, 0.0073767, 84.745, 3.5157, 4.48, -4.475613277589},
    {29, "Cu", 10.0, 5.0, 0.0057921, 84.843, 3.6030, 3.52, -3.519583146896},
    {45, "Rh", 13.0, 5.0, 0.0024612, 305.499, 3.7984, 5.78, -5.784242765216},
    {46, "Pd", 12.0, 6.0, 0.0032864, 148.205, 3.8813, 3.92, -3.916719132725},
    {47, "Ag", 11.0, 6.0, 0.0039450, 96.524, 4.0691, 2.97, -2.970491740121},
    {77, "Ir", 13.0, 6.0, 0.0037674, 224.815, 3.8344, 6.97, -6.968651001492},
    {78, "Pt", 11.0, 7.0, 0.0097894, 71.336, 3.9163, 5.86, -5.862273780458},
    {79, "Au", 11.0, 8.0, 0.0078052, 53.581, 4.0651, 3.83, -3.826503292017},
}};

const QscRow &find_row(int z) {
  for (const auto &row : kQscTable) {
    if (row.z == z) {
      return row;
    }
  }
  throw std::runtime_error("missing QSC row");
}

/// Heteronuclear dimer from Eqs. (1a)-(1c) and (2) of Qi et al. (1999): the
/// pair term and the density both use the mixed a_ab, with m_ab in the density.
void hetero_dimer_oracle(int za, int zb, double r, double *energy,
                         double *dEdr) {
  const QscRow &pa = find_row(za);
  const QscRow &pb = find_row(zb);
  const double eps_ab =
      za == zb ? pa.epsilon : std::sqrt(pa.epsilon * pb.epsilon);
  const double a_ab = 0.5 * (pa.a + pb.a);
  const double n_ab = 0.5 * (pa.n + pb.n);
  const double m_ab = 0.5 * (pa.m + pb.m);
  const double v_ab = eps_ab * std::pow(a_ab / r, n_ab);
  const double root_phi = std::sqrt(std::pow(a_ab / r, m_ab));
  const double emb_a = pa.c * pa.epsilon * root_phi;
  const double emb_b = pb.c * pb.epsilon * root_phi;
  *energy = v_ab - emb_a - emb_b;
  *dEdr = (-n_ab * v_ab + 0.5 * m_ab * (emb_a + emb_b)) / r;
}

std::vector<double> fcc_cell(int cells, double lattice, double *box) {
  const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  std::fill(box, box + 9, 0.0);
  box[0] = box[4] = box[8] = cells * lattice;
  std::vector<double> pos;
  for (int i = 0; i < cells * cells * cells; ++i) {
    for (const auto &b : basis) {
      pos.push_back((i % cells + b[0]) * lattice);
      pos.push_back((i / cells % cells + b[1]) * lattice);
      pos.push_back((i / (cells * cells) + b[2]) * lattice);
    }
  }
  return pos;
}

AtomMatrix load_sc_points(const std::string &path, double *epsilon, double *c,
                          double *a, double *nexp, double *mexp) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  in >> *epsilon >> *c >> *a >> *nexp >> *mexp;
  std::vector<double> xyz;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (in >> x >> y >> z) {
    xyz.push_back(x);
    xyz.push_back(y);
    xyz.push_back(z);
  }
  AtomMatrix pos(xyz.size() / 3, 3);
  for (std::size_t i = 0; i < xyz.size(); ++i) {
    pos(i / 3, i % 3) = xyz[i];
  }
  return pos;
}

} // namespace

TEST_CASE("QSC pure dimers at r = a0 match the closed-form identity",
          "[QSC]") {
  rgpot::QSCPot pot({.cutoff = 12.0, .periodic = false});
  for (const auto &row : kQscTable) {
    INFO(row.symbol);
    const AtomMatrix pos{{0.0, 0.0, 0.0}, {row.a, 0.0, 0.0}};
    const std::vector<int> types{row.z, row.z};
    auto [energy, forces, variance] = pot(pos, types, cube(40.0));
    (void)variance;
    const double want_e = row.epsilon * (1.0 - 2.0 * row.c);
    const double want_fx = row.epsilon * (row.m * row.c - row.n) / row.a;
    REQUIRE_THAT(energy, WithinAbs(want_e, kDimerTol));
    REQUIRE_THAT(forces(0, 0), WithinAbs(want_fx, kDimerTol));
    REQUIRE_THAT(forces(1, 0), WithinAbs(-want_fx, kDimerTol));
    REQUIRE_THAT(forces(0, 1), WithinAbs(0.0, kZeroForceTol));
    REQUIRE_THAT(forces(0, 2), WithinAbs(0.0, kZeroForceTol));
  }
}

TEST_CASE("QSC fcc crystals at 2*a0 reproduce the published cohesive energies",
          "[QSC]") {
  // Qi, Cagin, Kimura and Goddard, Phys. Rev. B 59, 3527 (1999), Table I and
  // Table II, and Ozdemir Kart et al., J. Non-Cryst. Solids 337, 101 (2004),
  // Table 1: cutoff 2*a0 includes 8 fcc shells (140 neighbours). A 5x5x5 fcc
  // supercell (500 atoms, box side 5*a0 > 4*a0) satisfies the minimum-image
  // condition for cutoff 2*a0.
  for (const auto &row : kQscTable) {
    INFO(row.symbol);
    double box[9] = {};
    const std::vector<double> pos = fcc_cell(5, row.a, box);
    const std::size_t n = pos.size() / 3;
    const std::vector<int> types(n, row.z);
    // Add kShellPad Angstrom so the 8th shell at exactly 2*a0 is not split by
    // binary64 rounding of sqrt(dx^2 + dy^2 + dz^2).
    const rgpot::QSCPot pot(
        {.cutoff = 2.0 * row.a + kShellPad, .periodic = true, .stress = true});
    std::vector<double> forces(3 * n, 0.0);
    rgpot::ForceOut out{};
    out.F = forces.data();
    pot.forceImpl({n, pos.data(), types.data(), box}, &out);
    const double e_per_atom = out.energy / static_cast<double>(n);
    REQUIRE_THAT(e_per_atom, WithinAbs(row.exact_2a_ecoh, kExactEcohTol));
    REQUIRE_THAT(-e_per_atom, WithinAbs(row.published_ecoh, kPublishedEcohTol));
    for (double f : forces) {
      REQUIRE_THAT(f, WithinAbs(0.0, kOracleTol));
    }
  }
}

TEST_CASE("QSC heteronuclear Cu-Ag and Cu-Ni dimers match alloy mixing rules",
          "[QSC]") {
  // Cu-Ag has half-integer n_ab = (10 + 11)/2 = 10.5 and m_ab = (5 + 6)/2 = 5.5,
  // and a_ab = (a_Cu + a_Ag)/2 in both the pair term and the density.
  rgpot::QSCPot pot({.cutoff = 10.0, .periodic = false});
  const std::pair<int, int> pairs[] = {
      {29, 47}, {29, 28}, {46, 78}, {47, 79}, {45, 77}};
  for (auto [za, zb] : pairs) {
    INFO("za=" << za << " zb=" << zb);
    const double r = 2.65;
    const AtomMatrix pos{{0.0, 0.0, 0.0}, {r, 0.0, 0.0}};
    const std::vector<int> types{za, zb};
    auto [energy, forces, variance] = pot(pos, types, cube(30.0));
    (void)variance;
    double want_e = 0.0;
    double dEdr = 0.0;
    hetero_dimer_oracle(za, zb, r, &want_e, &dEdr);
    REQUIRE_THAT(energy, WithinAbs(want_e, kOracleTol));
    REQUIRE_THAT(forces(0, 0), WithinAbs(dEdr, kOracleTol));
    REQUIRE_THAT(forces(1, 0), WithinAbs(-dEdr, kOracleTol));

    // Swapping atom order leaves the energy unchanged and negates the forces.
    const std::vector<int> swapped{zb, za};
    auto [e_swap, f_swap, v_swap] = pot(pos, swapped, cube(30.0));
    (void)v_swap;
    REQUIRE_THAT(e_swap, WithinAbs(energy, kDimerTol));
    REQUIRE_THAT(f_swap(0, 0), WithinAbs(forces(0, 0), kDimerTol));
  }
}

TEST_CASE("QSC reproduces Doye 10-8 cluster minima and matches SuttonChenPot",
          "[QSC]") {
  const char *files[] = {"CppCore/tests/data/sc108/pt38",
                         "CppCore/tests/data/sc108/pt75"};
  const double published[] = {-10117.2454, -20431.3452};
  for (int which = 0; which < 2; ++which) {
    double epsilon = 0.0;
    double c = 0.0;
    double a = 0.0;
    double nexp = 0.0;
    double mexp = 0.0;
    const AtomMatrix pos =
        load_sc_points(files[which], &epsilon, &c, &a, &nexp, &mexp);
    const std::vector<int> types(pos.rows(), 78);

    rgpot::QSCPot qsc({.cutoff = 40.0, .periodic = false});
    qsc.setElementParameter(78, nexp, mexp, epsilon, c, a);
    auto [e_qsc, f_qsc, v_qsc] = qsc(pos, types, cube(80.0));
    (void)v_qsc;
    REQUIRE_THAT(e_qsc, WithinAbs(published[which], kDoyeClusterTol));

    rgpot::SuttonChenPot sc({.epsilon = epsilon,
                             .c = c,
                             .a = a,
                             .n = static_cast<int>(nexp),
                             .m = static_cast<int>(mexp),
                             .cutoff = 40.0,
                             .periodic = false});
    auto [e_sc, f_sc, v_sc] = sc(pos, types, cube(80.0));
    (void)v_sc;
    REQUIRE(std::memcmp(&e_qsc, &e_sc, sizeof(double)) == 0);
    REQUIRE(std::memcmp(f_qsc.data(), f_sc.data(),
                        static_cast<std::size_t>(f_qsc.size()) *
                            sizeof(double)) == 0);
  }
}

TEST_CASE("QSC forces and Cauchy stress match finite differences on sheared alloy cells",
          "[QSC][stress]") {
  const double box[9] = {
      11.0, 0.0, 0.0, 0.8, 11.2, 0.0, -0.5, 0.6, 10.8};
  std::vector<int> z;
  std::vector<double> pos;
  const int species[4] = {29, 47, 28, 79};
  const double basis[4][3] = {
      {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  int idx = 0;
  for (int ix = 0; ix < 3; ++ix) {
    for (int iy = 0; iy < 3; ++iy) {
      for (int iz = 0; iz < 3; ++iz) {
        for (int b = 0; b < 4; ++b) {
          const double fx = (ix + basis[b][0]) / 3.0;
          const double fy = (iy + basis[b][1]) / 3.0;
          const double fz = (iz + basis[b][2]) / 3.0;
          for (int d = 0; d < 3; ++d) {
            const double r =
                fx * box[d] + fy * box[3 + d] + fz * box[6 + d];
            pos.push_back(r + 0.05 * std::sin(1.13 * idx + 0.7 * d));
          }
          z.push_back(species[idx % 4]);
          ++idx;
        }
      }
    }
  }
  const std::size_t n = z.size();
  const rgpot::QSCPot pot({.cutoff = 4.8, .periodic = true, .stress = true});
  std::vector<double> forces(3 * n, 0.0);
  rgpot::ForceOut out{};
  out.F = forces.data();
  pot.forceImpl({n, pos.data(), z.data(), box}, &out);
  REQUIRE(out.has_stress == 1);

  // Check forces on the first 8 atoms against 4th-order central differences.
  const double h = kFdForceStep;
  std::vector<double> scratch_f(3 * n, 0.0);
  auto energy_shifted = [&](std::size_t idx_c, double delta) {
    std::vector<double> shifted = pos;
    shifted[idx_c] += delta;
    rgpot::ForceOut so{};
    so.F = scratch_f.data();
    pot.forceImpl({n, shifted.data(), z.data(), box}, &so);
    return so.energy;
  };
  for (std::size_t a = 0; a < 8; ++a) {
    for (std::size_t d = 0; d < 3; ++d) {
      const std::size_t c_idx = 3 * a + d;
      const double d1 =
          -(energy_shifted(c_idx, h) - energy_shifted(c_idx, -h)) / (2.0 * h);
      const double d2 =
          -(energy_shifted(c_idx, 2.0 * h) - energy_shifted(c_idx, -2.0 * h)) /
          (4.0 * h);
      const double fd = (4.0 * d1 - d2) / 3.0;
      REQUIRE_THAT(forces[c_idx], WithinAbs(fd, kFdForceTol));
    }
  }

  // Check all 6 Voigt stress components against symmetric strain central
  // differences.
  const double volume = rgpot::cellVolume(box);
  const double eps = kFdStrainStep;
  for (int c = 0; c < 6; ++c) {
    const int p = kVoigt[c][0];
    const int q = kVoigt[c][1];
    auto eval_strain = [&](double s) {
      double e_mat[3][3] = {};
      if (p == q) {
        e_mat[p][p] = s;
      } else {
        e_mat[p][q] = 0.5 * s;
        e_mat[q][p] = 0.5 * s;
      }
      std::vector<double> spos(pos.size(), 0.0);
      for (std::size_t a = 0; a < n; ++a) {
        const double *r = &pos[3 * a];
        for (int i = 0; i < 3; ++i) {
          spos[3 * a + static_cast<std::size_t>(i)] =
              r[i] + r[0] * e_mat[0][i] + r[1] * e_mat[1][i] +
              r[2] * e_mat[2][i];
        }
      }
      double sbox[9] = {};
      for (int row = 0; row < 3; ++row) {
        const double *r = &box[3 * row];
        for (int i = 0; i < 3; ++i) {
          sbox[3 * row + i] = r[i] + r[0] * e_mat[0][i] + r[1] * e_mat[1][i] +
                              r[2] * e_mat[2][i];
        }
      }
      rgpot::ForceOut so{};
      so.F = scratch_f.data();
      pot.forceImpl({n, spos.data(), z.data(), sbox}, &so);
      return so.energy;
    };
    const double fd =
        (eval_strain(eps) - eval_strain(-eps)) / (2.0 * eps * volume);
    REQUIRE_THAT(out.stress[3 * p + q], WithinAbs(fd, kFdStressTol));
    REQUIRE_THAT(out.stress[3 * q + p],
                 WithinAbs(out.stress[3 * p + q], kZeroForceTol));
  }
}

TEST_CASE("QSC analytic Hessian-vector product matches 4th-order force differences",
          "[QSC][hvp]") {
  const double box[9] = {
      12.0, 0.0, 0.0, 0.6, 12.0, 0.0, -0.4, 0.5, 12.0};
  const std::vector<int> z = {28, 29, 46, 47, 78, 79};
  const std::vector<double> pos = {
      0.0,  0.0,  0.0,  2.55, 0.15, -0.10, 1.25, 2.30, 0.20,
      -1.2, 1.40, 2.10, 0.60, -1.8, 1.50,  2.10, 1.90, -1.60};
  std::vector<double> v(pos.size(), 0.0);
  for (std::size_t k = 0; k < v.size(); ++k) {
    v[k] = 0.35 * std::cos(1.1 * static_cast<double>(k) + 0.4);
  }

  rgpot::QSCPot pot({.cutoff = 6.0, .periodic = true});
  REQUIRE(pot.caps().analyticHvp);
  std::vector<double> hv(pos.size(), 0.0);
  const rgpot::ForceInput in{z.size(), pos.data(), z.data(), box};
  pot.hessianVectorImpl(in, v.data(), hv.data());
  REQUIRE(std::abs(hv[0]) > kHvpFloor);

  auto forces_at = [&](double step) {
    std::vector<double> shifted(pos.size());
    for (std::size_t k = 0; k < pos.size(); ++k) {
      shifted[k] = pos[k] + step * v[k];
    }
    std::vector<double> f(pos.size(), 0.0);
    rgpot::ForceOut out{};
    out.F = f.data();
    pot.forceImpl({z.size(), shifted.data(), z.data(), box}, &out);
    return f;
  };
  const double h = kFdForceStep;
  const auto fp1 = forces_at(h);
  const auto fm1 = forces_at(-h);
  const auto fp2 = forces_at(2.0 * h);
  const auto fm2 = forces_at(-2.0 * h);
  for (std::size_t k = 0; k < pos.size(); ++k) {
    const double d1 = -(fp1[k] - fm1[k]) / (2.0 * h);
    const double d2 = -(fp2[k] - fm2[k]) / (4.0 * h);
    const double richardson = (4.0 * d1 - d2) / 3.0;
    REQUIRE_THAT(
        hv[k], WithinAbs(richardson, kHvpRelTol * (1.0 + std::abs(richardson))));
  }

  // HessianVectorCache dispatches HvpScheme::Auto to hessianVectorImpl with
  // zero force evaluations.
  rgpot::HessianVectorCache cache;
  std::vector<double> hv_cached(pos.size(), 0.0);
  cache.apply(pot, in, v.data(), hv_cached.data(), 0.0, rgpot::HvpScheme::Auto);
  REQUIRE(cache.counts().products == 1);
  REQUIRE(cache.counts().analytic == 1);
  REQUIRE(cache.counts().forceCalls == 0);
  for (std::size_t k = 0; k < pos.size(); ++k) {
    REQUIRE(hv_cached[k] == hv[k]);
  }
}

TEST_CASE("QSC validates configuration, element parameters and force inputs",
          "[QSC]") {
  REQUIRE_THROWS_AS(rgpot::QSCPot({.cutoff = 0.0, .periodic = true}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(rgpot::QSCPot({.verlet_skin = -0.1}),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
      rgpot::QSCPot({.cutoff = std::numeric_limits<double>::quiet_NaN()}),
      std::invalid_argument);

  rgpot::QSCPot pot;
  const uint64_t k0 = pot.paramsKey();
  REQUIRE(k0 != 0);
  pot.setElementParameter(29, 10.0, 5.0, 0.006, 84.843, 3.52);
  REQUIRE(pot.paramsKey() != k0);

  REQUIRE_THROWS_AS(pot.setElementParameter(0, 10.0, 5.0, 0.01, 50.0, 3.5),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setElementParameter(29, 0.5, 5.0, 0.01, 50.0, 3.5),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setElementParameter(29, 10.0, 0.5, 0.01, 50.0, 3.5),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setElementParameter(29, 10.0, 5.0, 0.0, 50.0, 3.5),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setElementParameter(29, 10.0, 5.0, 0.01, 0.0, 3.5),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(pot.setElementParameter(29, 10.0, 5.0, 0.01, 50.0, 0.0),
                    std::invalid_argument);

  const double box[9] = {20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0};
  const double pos[6] = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0};
  const int bad_z[2] = {29, 14}; // Si (14) is not in the default Q-SC table.
  double forces[6] = {};
  rgpot::ForceOut out{};
  out.F = forces;
  REQUIRE_THROWS_AS(pot.forceImpl({2, pos, bad_z, box}, &out),
                    std::invalid_argument);

  // Coincident atoms throw std::domain_error.
  const double coincident[6] = {1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
  const int cu_z[2] = {29, 29};
  REQUIRE_THROWS_AS(pot.forceImpl({2, coincident, cu_z, box}, &out),
                    std::domain_error);
  double hv[6] = {};
  const double dir[6] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  REQUIRE_THROWS_AS(
      pot.hessianVectorImpl({2, coincident, cu_z, box}, dir, hv),
      std::domain_error);
}
