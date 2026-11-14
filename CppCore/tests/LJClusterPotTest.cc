// MIT License
// Copyright 2023--present rgpot developers
//
// Free-boundary Lennard-Jones cluster checks: the eOn energy pin, agreement
// with the periodic LJPot on an isolated cluster, and cell independence.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <catch2/catch_all.hpp>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using rgpot::types::AtomMatrix;

namespace {

/// eOn's neb_morse reactant geometry (13 hydrogens, client/unit_tests/data/
/// systems/neb_morse/reactant.con), inlined because rgpot reads no .con.
AtomMatrix nebMorseCluster() {
  return AtomMatrix{
      {50.594227, 52.017165, 52.277482}, {51.578540, 52.642148, 51.195222},
      {51.243758, 52.919086, 52.218383}, {50.490127, 52.736378, 51.417663},
      {51.522643, 50.884535, 51.656125}, {50.927263, 51.743087, 51.272643},
      {51.240676, 50.960436, 50.573042}, {51.275727, 52.061877, 50.284160},
      {50.434110, 50.976423, 51.879062}, {51.695007, 51.919472, 52.038680},
      {51.363596, 52.199101, 53.064923}, {52.017483, 51.639074, 51.008389},
      {51.187843, 51.165217, 52.678225}};
}

/// The cell that ships with the geometry above.
std::array<std::array<double, 3>, 3> nebMorseCell() {
  return {
      {{101.942400, 0.0, 0.0}, {0.0, 103.142600, 0.0}, {0.0, 0.0, 102.605500}}};
}

double maxForceNorm(const AtomMatrix &forces) {
  double worst = 0.0;
  for (size_t i = 0; i < forces.rows(); ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  return worst;
}

} // namespace

TEST_CASE("LJClusterPot energy on the neb_morse cluster", "[LJClusterPot]") {
  rgpot::LJClusterPot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);

  auto [energy, forces, variance] = pot(positions, atmtypes, nebMorseCell());
  (void)variance;

  // eOn reference (client/unit_tests/PotTest.cpp, "LJCluster energy matches
  // SVN"), carried over with its epsilon.
  REQUIRE_THAT(energy, WithinRel(-39.965379, 1e-4));
  // Tighter pin from an independent double-precision evaluation of this
  // kernel; eOn's LJ case pins the same maximum force to 1e-2.
  REQUIRE_THAT(energy, WithinRel(-39.9653513626179, 1e-9));
  REQUIRE_THAT(maxForceNorm(forces), WithinRel(0.00470419517301762, 1e-9));
}

TEST_CASE("LJClusterPot matches LJPot on an isolated cluster",
          "[LJClusterPot]") {
  // Same 12-6 kernel and cutoff shift; the cell is wide enough that the
  // minimum image convention leaves LJPot with the identical pair set.
  rgpot::LJClusterPot cluster;
  rgpot::LJPot periodic;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);
  const auto cell = nebMorseCell();

  auto [e_cluster, f_cluster, v_cluster] = cluster(positions, atmtypes, cell);
  auto [e_periodic, f_periodic, v_periodic] =
      periodic(positions, atmtypes, cell);
  (void)v_cluster;
  (void)v_periodic;

  REQUIRE_THAT(e_cluster, WithinRel(e_periodic, 1e-12));
  for (size_t i = 0; i < f_cluster.rows(); ++i) {
    for (size_t j = 0; j < f_cluster.cols(); ++j) {
      REQUIRE_THAT(f_cluster(i, j), WithinAbs(f_periodic(i, j), 1e-12));
    }
  }
}

TEST_CASE("LJClusterPot ignores the cell", "[LJClusterPot]") {
  // Free boundaries: a cell narrower than the cutoff must not fold any pair
  // vector, so the energy is unchanged.
  rgpot::LJClusterPot pot;
  const auto positions = nebMorseCluster();
  const std::vector<int> atmtypes(13, 1);
  const std::array<std::array<double, 3>, 3> tight{
      {{5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0}}};

  auto [e_wide, f_wide, v_wide] = pot(positions, atmtypes, nebMorseCell());
  auto [e_tight, f_tight, v_tight] = pot(positions, atmtypes, tight);
  (void)v_wide;
  (void)v_tight;

  REQUIRE_THAT(e_tight, WithinRel(e_wide, 1e-12));
  REQUIRE_FALSE(pot.caps().periodic);
}

namespace {

void useThreads(int n) {
  const auto text = std::to_string(n);
  setenv("OMP_NUM_THREADS", text.c_str(), 1);
#if defined(_OPENMP)
  omp_set_num_threads(n);
#endif
}

/// Free-boundary pair sum. This is the one-thread cell walk: increasing
/// i, then increasing j, raw coordinate differences, no image fold.
void rawCluster(const double *pos, int n, double cutoff, double &energy, double *force) {
  const double cut2 = cutoff * cutoff;
  const double psi2 = 1.0;
  const double four = 4.0;
  const double a_cut = std::pow(1.0 / cutoff, 6.0);
  const double shift = 4.0 * a_cut * (a_cut - 1.0);
  energy = 0.0;
  for (int i = 0; i < 3 * n; ++i) {
    force[i] = 0.0;
  }
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double dx = pos[3 * i] - pos[3 * j];
      const double dy = pos[3 * i + 1] - pos[3 * j + 1];
      const double dz = pos[3 * i + 2] - pos[3 * j + 2];
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (!(r2 > 0.0) || r2 > cut2) {
        continue;
      }
      const double inv = 1.0 / r2;
      const double sr2 = psi2 * inv;
      const double a = (sr2 * sr2) * sr2;
      const double b = four * a;
      energy += b * (a - 1.0) - shift;
      const double fscale = ((6.0 * b) * inv) * ((2.0 * a) - 1.0);
      const double fx = fscale * dx;
      const double fy = fscale * dy;
      const double fz = fscale * dz;
      force[3 * i] += fx;
      force[3 * i + 1] += fy;
      force[3 * i + 2] += fz;
      force[3 * j] -= fx;
      force[3 * j + 1] -= fy;
      force[3 * j + 2] -= fz;
    }
  }
}

} // namespace

TEST_CASE("LJClusterPot raw differences match across cells and threads",
          "[LJClusterPot][threads]") {
  // The span exceeds the cutoff, so the call uses the stored list rather
  // than the all-pairs kernel. One atom sits on each side of the origin,
  // and the pair at 2.5 A would change under a 4 A periodic fold.
  const double pos[] = {0.0, 0.0, 0.0, 2.5, 0.0, 0.0, -1.5, 0.2, 0.4, 18.0, 1.0, -0.5};
  const int n = 4;
  const double cutoff = 3.0;
  rgpot::LJClusterPot pot{rgpot::LJClusterConfig{.cutoff = cutoff}};
  const std::vector<int> types(static_cast<std::size_t>(n), 1);
  AtomMatrix positions(static_cast<std::size_t>(n), 3);
  for (int i = 0; i < 3 * n; ++i) {
    positions.data()[i] = pos[i];
  }
  const std::array<std::array<double, 3>, 3> wide{
      {{40.0, 0.0, 0.0}, {0.0, 40.0, 0.0}, {0.0, 0.0, 40.0}}};
  const std::array<std::array<double, 3>, 3> tight{
      {{4.0, 0.0, 0.0}, {0.0, 4.0, 0.0}, {0.0, 0.0, 4.0}}};

  double ref_e = 0.0;
  double ref_f[12] = {};
  rawCluster(pos, n, cutoff, ref_e, ref_f);

  useThreads(1);
  auto [e1, f1, v1] = pot(positions, types, wide);
  (void)v1;
  REQUIRE(e1 == ref_e);
  for (int i = 0; i < 3 * n; ++i) {
    REQUIRE(f1.data()[i] == ref_f[i]);
  }
  // A periodic-sized cell must not fold. One thread stays on the free walk.
  auto [e_tight, f_tight, v_tight] = pot(positions, types, tight);
  (void)v_tight;
  REQUIRE(e_tight == e1);
  for (int i = 0; i < 3 * n; ++i) {
    REQUIRE(f_tight.data()[i] == f1.data()[i]);
  }

  for (int threads : {2, 4, 8}) {
    useThreads(threads);
    auto [e_n, f_n, v_n] = pot(positions, types, tight);
    (void)v_n;
    REQUIRE(e_n == e1);
    for (int i = 0; i < 3 * n; ++i) {
      REQUIRE(f_n.data()[i] == f1.data()[i]);
    }
    auto [e_w, f_w, v_w] = pot(positions, types, wide);
    (void)v_w;
    REQUIRE(e_w == e1);
    for (int i = 0; i < 3 * n; ++i) {
      REQUIRE(f_w.data()[i] == f1.data()[i]);
    }
  }
  useThreads(1);
}

TEST_CASE("a periodic boundary pair stays on the folded walk",
          "[LJClusterPot][threads][stress]") {
  // x = 0.1 and x = L-0.1. The minimum image is 0.2 and the raw difference
  // is L-0.2. The search is shorter than half the box, and the pair still
  // crosses the boundary. LJ keeps free boundaries, so every thread count
  // matches the one-thread walk, including the untouched stress slot.
  const double side = 10.0;
  const double pos[] = {0.1, 0.0, 0.0, side - 0.1, 0.0, 0.0, 0.3, 0.1, 0.0};
  const int n = 3;
  const double cutoff = 0.5;
  const std::vector<int> types(static_cast<std::size_t>(n), 1);
  const double box[9] = {side, 0, 0, 0, side, 0, 0, 0, side};
  auto run = [&](int threads, double *force, double &energy, double *stress, int &has_stress) {
    useThreads(threads);
    rgpot::LJClusterPot pot{rgpot::LJClusterConfig{.cutoff = cutoff}};
    rgpot::ForceOut out{force, 0.0, 0.0, {}, 0};
    pot.forceImpl(rgpot::ForceInput{static_cast<std::size_t>(n), pos, types.data(), box}, &out);
    energy = out.energy;
    has_stress = out.has_stress;
    std::copy(out.stress, out.stress + 9, stress);
  };
  double f1[9] = {};
  double s1[9] = {};
  double e1 = 0.0;
  int h1 = 0;
  run(1, f1, e1, s1, h1);
  REQUIRE(e1 != 0.0);
  for (int threads : {2, 4, 8}) {
    double fn[9] = {};
    double sn[9] = {};
    double en = 0.0;
    int hn = 0;
    run(threads, fn, en, sn, hn);
    REQUIRE(en == e1);
    REQUIRE(hn == h1);
    for (int i = 0; i < 9; ++i) {
      REQUIRE(fn[i] == f1[i]);
      REQUIRE(sn[i] == s1[i]);
    }
  }
  useThreads(1);
}

TEST_CASE("LJ cluster chunk sums match at every thread count", "[LJClusterPot][threads]") {
  // Several atoms share a chunk, and 3 is not a divisor of the chunk count.
  // Energy and every force component must match the one-thread call.
  static const double basis[4][3] = {{0, 0, 0}, {0, 0.5, 0.5}, {0.5, 0, 0.5}, {0.5, 0.5, 0}};
  auto bits = [](double x) {
    std::uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
  };
  for (int n : {256, 1024, 4096}) {
    int cells = n / 4;
    int nx = 1;
    while (nx * nx * nx < cells) {
      ++nx;
    }
    const int ny = nx;
    const int nz = std::max(1, cells / (nx * ny));
    AtomMatrix pos(static_cast<std::size_t>(n), 3);
    int atom = 0;
    for (int ix = 0; ix < nx && atom < n; ++ix) {
      for (int iy = 0; iy < ny && atom < n; ++iy) {
        for (int iz = 0; iz < nz && atom < n; ++iz) {
          for (int k = 0; k < 4 && atom < n; ++k, ++atom) {
            pos(static_cast<std::size_t>(atom), 0) = (ix + basis[k][0]) * 3.5 + 0.01 * std::sin(atom);
            pos(static_cast<std::size_t>(atom), 1) = (iy + basis[k][1]) * 3.5;
            pos(static_cast<std::size_t>(atom), 2) = (iz + basis[k][2]) * 3.5;
          }
        }
      }
    }
    const std::array<std::array<double, 3>, 3> box{
        {{nx * 3.5, 0.0, 0.0}, {0.0, ny * 3.5, 0.0}, {0.0, 0.0, nz * 3.5}}};
    const std::vector<int> types(static_cast<std::size_t>(n), 18);
    rgpot::LJClusterPot pot;
    useThreads(1);
    auto [e1, f1, v1] = pot(pos, types, box);
    (void)v1;
    const double *ref = f1.data();
    for (int threads : {2, 3, 4, 8}) {
      useThreads(threads);
      auto [en, fn, vn] = pot(pos, types, box);
      (void)vn;
      REQUIRE(bits(en) == bits(e1));
      for (int i = 0; i < 3 * n; ++i) {
        REQUIRE(bits(fn.data()[i]) == bits(ref[i]));
      }
    }
  }
  useThreads(1);
}

TEST_CASE("LJClusterPot parameter fingerprint tracks the config",
          "[LJClusterPot]") {
  const rgpot::LJClusterPot defaults;
  const rgpot::LJClusterPot wider{rgpot::LJClusterConfig{.cutoff = 20.0}};
  const rgpot::LJClusterPot sameAsDefaults{rgpot::LJClusterConfig{}};

  REQUIRE(defaults.paramsKey() == sameAsDefaults.paramsKey());
  REQUIRE(defaults.paramsKey() != wider.paramsKey());
  REQUIRE(defaults.get_type() == rgpot::PotType::LJCluster);
  // Identical numbers hash identically; the result cache keeps the two
  // kernels apart through PotType, not through paramsKey.
  REQUIRE(defaults.paramsKey() == rgpot::LJPot{}.paramsKey());
}

TEST_CASE("LJPot stress matches a coordinate difference", "[LJPot]") {
  rgpot::LJPot pot;
  REQUIRE(pot.caps().stress);

  AtomMatrix pos{{0.0, 0.0, 0.0}, {1.3, 0.4, -0.25}};
  const std::vector<int> types{1, 1};
  // Row-major cell, large enough that the pair is the minimum image.
  const double box[9] = {12.0, 0.0, 0.0, 0.4, 11.0, 0.0, -0.2, 0.3, 10.0};
  double forces[6] = {};
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  pot.forceImpl(rgpot::ForceInput{2, pos.data(), types.data(), box}, &out);
  REQUIRE(out.has_stress == 1);

  const double volume = std::abs(box[0] * (box[4] * box[8] - box[5] * box[7]) -
                                 box[1] * (box[3] * box[8] - box[5] * box[6]) +
                                 box[2] * (box[3] * box[7] - box[4] * box[6]));
  constexpr double step = 1e-6;
  const int rows[6] = {0, 1, 2, 1, 0, 0};
  const int cols[6] = {0, 1, 2, 2, 2, 1};
  for (int comp = 0; comp < 6; ++comp) {
    const int row = rows[comp];
    const int col = cols[comp];
    auto energyAt = [&](double eps) {
      AtomMatrix shifted = pos;
      double strained[9];
      std::copy(box, box + 9, strained);
      for (int atom = 0; atom < 2; ++atom) {
        shifted(atom, col) += pos(atom, row) * eps;
      }
      for (int k = 0; k < 3; ++k) {
        strained[k * 3 + col] += box[k * 3 + row] * eps;
      }
      double f2[6] = {};
      rgpot::ForceOut sample{f2, 0.0, 0.0, {}, 0};
      pot.forceImpl(
          rgpot::ForceInput{2, shifted.data(), types.data(), strained},
          &sample);
      return sample.energy;
    };
    const double derivative = (energyAt(step) - energyAt(-step)) / (2.0 * step);
    const double expected = derivative / volume;
    const double got = out.stress[row * 3 + col];
    REQUIRE_THAT(got, WithinAbs(expected, 1e-6));
  }
}

namespace {
template <class Potential>
void checkPairStressByStrain(const Potential &pot, const AtomMatrix &positions,
                             const std::vector<int> &types) {
  REQUIRE(pot.caps().stress);
  const std::array<double, 9> box{20.0, 0.0, 0.0, 0.0, 21.0,
                                  0.0,  0.0, 0.0, 22.0};
  constexpr double volume = 20.0 * 21.0 * 22.0;
  constexpr double step = 1e-6;
  AtomMatrix forces(positions.rows(), 3);
  rgpot::ForceOut result{forces.data(), 0.0, 0.0, {}, 0};
  const rgpot::ForceInput input{static_cast<size_t>(positions.rows()),
                                positions.data(), types.data(), box.data()};
  pot.forceImpl(input, &result);
  REQUIRE(result.has_stress == 1);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const auto energyAt = [&](double strain) {
        AtomMatrix moved = positions;
        auto cell = box;
        for (int atom = 0; atom < positions.rows(); ++atom) {
          moved(atom, col) += strain * positions(atom, row);
        }
        for (int axis = 0; axis < 3; ++axis) {
          cell[3 * axis + col] += strain * box[3 * axis + row];
        }
        AtomMatrix sample_forces(positions.rows(), 3);
        rgpot::ForceOut sample{sample_forces.data(), 0.0, 0.0, {}, 0};
        pot.forceImpl({static_cast<size_t>(positions.rows()), moved.data(),
                       types.data(), cell.data()},
                      &sample);
        return sample.energy;
      };
      const double expected =
          (energyAt(step) - energyAt(-step)) / (2.0 * step * volume);
      CAPTURE(row, col, expected);
      REQUIRE_THAT(result.stress[3 * row + col], WithinAbs(expected, 1e-8));
    }
  }
  rgpot::ForceOut repeated{forces.data(), 0.0, 0.0, {}, 0};
  pot.forceImpl(input, &repeated);
  REQUIRE(repeated.energy == result.energy);
  REQUIRE(repeated.has_stress == 1);
  for (size_t i = 0; i < 9; ++i) {
    REQUIRE(repeated.stress[i] == result.stress[i]);
  }
}
} // namespace

TEST_CASE(
    "Pair stresses differentiate the actual truncated and switched energy",
    "[stress][LJPot][MorsePot][ZBLPot]") {
  const std::vector<int> types{6, 8};
  SECTION("periodic Lennard-Jones pair crosses the cell boundary") {
    rgpot::LJPot pot;
    AtomMatrix positions{{19.1, 0.1, -0.2}, {0.5, 0.4, 0.3}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Lennard-Jones switch contributes to stress") {
    rgpot::LJPot pot(rgpot::LJConfig{.cutoff = 3.0, .switch_width = 0.8});
    AtomMatrix positions{{0.0, 0.0, 0.0}, {2.6, 0.2, -0.1}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Morse shifted energy") {
    rgpot::MorsePot pot;
    AtomMatrix positions{{0.0, 0.0, 0.0}, {3.1, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("Morse switch contributes to stress") {
    rgpot::MorsePot pot(rgpot::MorseConfig{.cutoff = 5.0, .switch_width = 1.0});
    AtomMatrix positions{{0.0, 0.0, 0.0}, {4.3, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
  SECTION("ZBL screened pair") {
    rgpot::ZBLPot pot;
    AtomMatrix positions{{0.0, 0.0, 0.0}, {1.3, 0.3, -0.2}};
    checkPairStressByStrain(pot, positions, types);
  }
}

TEST_CASE("Pair stress is cleared when the configuration has no pair",
          "[stress][LJPot][MorsePot][ZBLPot]") {
  const int type = 6;
  const double position[3] = {0.0, 0.0, 0.0};
  const double box[9] = {20.0, 0.0, 0.0, 0.0, 21.0, 0.0, 0.0, 0.0, 22.0};
  double forces[3] = {};
  const auto check = [&](const auto &pot) {
    rgpot::ForceOut out{
        forces, 1.0, 0.0, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, 0};
    pot.forceImpl({1, position, &type, box}, &out);
    REQUIRE(out.has_stress == 1);
    REQUIRE(out.energy == 0.0);
    for (double component : out.stress) {
      REQUIRE(component == 0.0);
    }
  };
  check(rgpot::LJPot{});
  check(rgpot::MorsePot{});
  check(rgpot::ZBLPot{});
}
