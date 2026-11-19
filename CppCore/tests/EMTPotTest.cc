// MIT License
// Copyright 2023--present rgpot developers
//
// Copper FCC cluster pin from eOn EMTCuTest / point_emt_cu.dat.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/ForceStructs.hpp"
#include "rgpot/Slices.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("EMT energy matches the copper cluster pin", "[emt]") {
  const std::vector<std::array<double, 3>> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  rgpot::types::AtomMatrix positions(4, 3);
  std::vector<int> types(4, 29);
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  rgpot::EMTPot pot(rgpot::EMTConfig{.rasmussen = false});
  const auto [energy, forces, variance] = pot(positions, types, box);
  REQUIRE_THAT(energy, WithinAbs(5.129167, 5e-4));
  double worst = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double n2 = forces(i, 0) * forces(i, 0) +
                      forces(i, 1) * forces(i, 1) + forces(i, 2) * forces(i, 2);
    worst = std::max(worst, std::sqrt(n2));
  }
  REQUIRE_THAT(worst, WithinAbs(1.914263, 2e-3));
  REQUIRE(pot.config().rasmussen == false);
  REQUIRE(pot.paramsKey() != 0);
}

TEST_CASE("EMT reads the species again when only the species change", "[emt]") {
  rgpot::types::AtomMatrix positions(4, 3);
  const std::array<std::array<double, 3>, 4> xyz = {{
      {8.1925, 8.1925, 8.1925},
      {10.0, 10.0, 8.1925},
      {10.0, 8.1925, 10.0},
      {8.1925, 10.0, 10.0},
  }};
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      positions(i, k) = xyz[static_cast<size_t>(i)][static_cast<size_t>(k)];
    }
  }
  const std::array<std::array<double, 3>, 3> box = {{
      {20.0, 0.0, 0.0},
      {0.0, 20.0, 0.0},
      {0.0, 0.0, 20.0},
  }};
  const std::vector<int> copper(4, 29);
  const std::vector<int> mixed = {29, 79, 29, 79};

  rgpot::EMTPot reused;
  (void)reused(positions, copper, box);
  const auto [e_reused, f_reused, v_reused] = reused(positions, mixed, box);

  rgpot::EMTPot fresh;
  const auto [e_fresh, f_fresh, v_fresh] = fresh(positions, mixed, box);

  REQUIRE_THAT(e_reused, WithinAbs(e_fresh, 1e-12));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(f_reused(i, k), WithinAbs(f_fresh(i, k), 1e-12));
    }
  }
}

namespace {

struct ThreadGuard {
  std::string saved;
  bool had = false;

  ThreadGuard() {
    if (const char *v = std::getenv("OMP_NUM_THREADS")) {
      saved = v;
      had = true;
    }
  }

  ~ThreadGuard() { set(had ? saved.c_str() : nullptr); }

  static void set(const char *v) {
#if defined(_WIN32)
    _putenv_s("OMP_NUM_THREADS", v == nullptr ? "" : v);
#else
    if (v == nullptr || v[0] == '\0') {
      unsetenv("OMP_NUM_THREADS");
    } else {
      setenv("OMP_NUM_THREADS", v, 1);
    }
#endif
  }
};

std::uint64_t f64Bits(double v) {
  std::uint64_t out = 0;
  std::memcpy(&out, &v, sizeof(out));
  return out;
}

struct Eval {
  double energy = 0.0;
  std::vector<double> forces;
  std::array<double, 9> stress{};
  int has_stress = 0;
};

Eval runEval(rgpot::EMTPot &pot, const std::vector<double> &pos,
             const std::vector<int> &z, const std::array<double, 9> &box,
             bool stress = false) {
  Eval outv;
  outv.forces.assign(pos.size(), 0.0);
  rgpot::ForceOut out{};
  out.F = outv.forces.data();
  std::fill(std::begin(out.stress), std::end(out.stress), 0.0);
  rgpot::ForceInput in{z.size(), pos.data(), z.data(), box.data()};
  in.compute_stress = stress;
  pot.forceImpl(in, &out);
  outv.energy = out.energy;
  outv.has_stress = out.has_stress;
  std::copy(std::begin(out.stress), std::end(out.stress), outv.stress.begin());
  return outv;
}

std::size_t countBitDiffs(const Eval &a, const Eval &b) {
  std::size_t diffs = 0;
  diffs += (f64Bits(a.energy) == f64Bits(b.energy)) ? 0 : 1;
  diffs += (a.has_stress == b.has_stress) ? 0 : 1;
  for (std::size_t k = 0; k < 9; ++k) {
    diffs += (f64Bits(a.stress[k]) == f64Bits(b.stress[k])) ? 0 : 1;
  }
  for (std::size_t k = 0; k < a.forces.size(); ++k) {
    diffs += (f64Bits(a.forces[k]) == f64Bits(b.forces[k])) ? 0 : 1;
  }
  return diffs;
}

struct FccBox {
  std::vector<double> pos;
  std::vector<int> z;
  std::array<double, 9> box{};
};

FccBox fccCu(int side, double lattice) {
  static constexpr double kBasis[4][3] = {
      {0.0, 0.0, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}, {0.5, 0.5, 0.0}};
  FccBox out;
  const double L = static_cast<double>(side) * lattice;
  out.box = {L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L};
  for (int ix = 0; ix < side; ++ix) {
    for (int iy = 0; iy < side; ++iy) {
      for (int iz = 0; iz < side; ++iz) {
        for (const auto &b : kBasis) {
          const double fx =
              (static_cast<double>(ix) + b[0]) / static_cast<double>(side);
          const double fy =
              (static_cast<double>(iy) + b[1]) / static_cast<double>(side);
          const double fz =
              (static_cast<double>(iz) + b[2]) / static_cast<double>(side);
          const std::size_t idx = out.z.size();
          const double jx =
              0.01 * lattice * std::sin(0.23 * static_cast<double>(idx) + 0.1);
          const double jy =
              0.01 * lattice * std::cos(0.31 * static_cast<double>(idx) + 0.2);
          const double jz =
              0.01 * lattice * std::sin(0.29 * static_cast<double>(idx) + 0.3);
          out.pos.push_back(fx * L + jx);
          out.pos.push_back(fy * L + jy);
          out.pos.push_back(fz * L + jz);
          out.z.push_back(29);
        }
      }
    }
  }
  return out;
}

std::vector<double> shifted(const std::vector<double> &pos, double step) {
  std::vector<double> out = pos;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] += step * std::sin(0.71 * static_cast<double>(i) + 0.4);
  }
  return out;
}

void rightTransform(std::vector<double> &pos, std::array<double, 9> &box,
                    const std::array<double, 9> &map) {
  auto transform = [&](double *v) {
    const std::array<double, 3> old{v[0], v[1], v[2]};
    for (int b = 0; b < 3; ++b) {
      v[b] = old[0] * map[static_cast<std::size_t>(b)] +
             old[1] * map[static_cast<std::size_t>(3 + b)] +
             old[2] * map[static_cast<std::size_t>(6 + b)];
    }
  };
  for (std::size_t i = 0; i < pos.size(); i += 3) {
    transform(pos.data() + i);
  }
  for (int i = 0; i < 3; ++i) {
    transform(box.data() + 3 * i);
  }
}

} // namespace

TEST_CASE("EMT stress is opt-in and matches triclinic strain",
          "[emt][stress]") {
  auto system = fccCu(4, 3.615);
  const std::array<double, 9> shear = {1.0,  0.12, 0.03,  0.02, 0.96,
                                       0.08, 0.04, -0.03, 1.05};
  rightTransform(system.pos, system.box, shear);

  rgpot::EMTPot plain_pot;
  rgpot::EMTPot stress_pot;
  const Eval plain = runEval(plain_pot, system.pos, system.z, system.box);
  const Eval analytic =
      runEval(stress_pot, system.pos, system.z, system.box, true);
  REQUIRE(plain.has_stress == 0);
  REQUIRE(analytic.has_stress == 1);
  REQUIRE(f64Bits(plain.energy) == f64Bits(analytic.energy));
  REQUIRE(plain.forces.size() == analytic.forces.size());
  for (std::size_t i = 0; i < plain.forces.size(); ++i) {
    REQUIRE(f64Bits(plain.forces[i]) == f64Bits(analytic.forces[i]));
  }
  CHECK(analytic.stress[1] == analytic.stress[3]);
  CHECK(analytic.stress[2] == analytic.stress[6]);
  CHECK(analytic.stress[5] == analytic.stress[7]);

  static constexpr int slots[6] = {0, 4, 8, 5, 2, 1};
  double target_error = 0.0;
  double best_other = std::numeric_limits<double>::infinity();
  for (double step : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7}) {
    double max_error = 0.0;
    double scale = 0.0;
    for (int component = 0; component < 6; ++component) {
      const int a = component < 3 ? component : (component == 3 ? 1 : 0);
      const int b = component < 3 ? component
                                  : (component == 3   ? 2
                                     : component == 4 ? 2
                                                      : 1);
      std::array<double, 9> plus_map{1, 0, 0, 0, 1, 0, 0, 0, 1};
      std::array<double, 9> minus_map = plus_map;
      const double weight = a == b ? step : 0.5 * step;
      plus_map[static_cast<std::size_t>(3 * a + b)] += weight;
      minus_map[static_cast<std::size_t>(3 * a + b)] -= weight;
      if (a != b) {
        plus_map[static_cast<std::size_t>(3 * b + a)] += weight;
        minus_map[static_cast<std::size_t>(3 * b + a)] -= weight;
      }
      auto plus_pos = system.pos;
      auto minus_pos = system.pos;
      auto plus_box = system.box;
      auto minus_box = system.box;
      rightTransform(plus_pos, plus_box, plus_map);
      rightTransform(minus_pos, minus_box, minus_map);
      rgpot::EMTPot plus_pot;
      rgpot::EMTPot minus_pot;
      const double ep = runEval(plus_pot, plus_pos, system.z, plus_box).energy;
      const double em =
          runEval(minus_pot, minus_pos, system.z, minus_box).energy;
      const double volume = std::abs(
          system.box[0] *
              (system.box[4] * system.box[8] - system.box[5] * system.box[7]) -
          system.box[1] *
              (system.box[3] * system.box[8] - system.box[5] * system.box[6]) +
          system.box[2] *
              (system.box[3] * system.box[7] - system.box[4] * system.box[6]));
      const double fd = (ep - em) / (2.0 * step * volume);
      const double value = analytic.stress[slots[component]];
      max_error = std::max(max_error, std::abs(value - fd));
      scale = std::max(scale, std::abs(value));
    }
    const double relative = max_error / std::max(scale, 1e-12);
    if (step == 1e-5) {
      target_error = relative;
    } else {
      best_other = std::min(best_other, relative);
    }
  }
  std::cout << "STRESS_ERROR emt " << std::setprecision(12) << target_error
            << '\n';
  CHECK(target_error <= 2e-7);
  CHECK(target_error <= best_other * 2.0 + 1e-12);
}

TEST_CASE("EMT OpenMP force path is bit-identical across threads",
          "[emt][threads]") {
  ThreadGuard guard;
  rgpot::EMTPot sanity;

  const auto small = fccCu(12, 3.615 / std::sqrt(2.0));
  REQUIRE(small.z.size() < rgpot::kTeamMinAtoms);
  const auto small_reuse = shifted(small.pos, 0.02);
  const auto small_rebuild = shifted(small.pos, 0.35);

  ThreadGuard::set("1");
  auto ref_small = runEval(sanity, small.pos, small.z, small.box);
  auto ref_small_reuse = runEval(sanity, small_reuse, small.z, small.box);
  auto ref_small_rebuild = runEval(sanity, small_rebuild, small.z, small.box);
  REQUIRE(ref_small.has_stress == 0);

  for (const char *th : {"2", "4", "8"}) {
    INFO("small OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    const auto before = rgpot::teamsOpened();
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, small.pos, small.z, small.box),
                        ref_small) == 0);
    CHECK(countBitDiffs(runEval(pot, small_reuse, small.z, small.box),
                        ref_small_reuse) == 0);
    CHECK(countBitDiffs(runEval(pot, small_rebuild, small.z, small.box),
                        ref_small_rebuild) == 0);
    CHECK(rgpot::teamsOpened() == before);
  }

  const auto large = fccCu(13, 3.615 / std::sqrt(2.0));
  REQUIRE(large.z.size() >= rgpot::kTeamMinAtoms);
  const auto large_reuse = shifted(large.pos, 0.02);
  const auto large_rebuild = shifted(large.pos, 0.35);

  ThreadGuard::set("1");
  rgpot::EMTPot serial;
  auto ref_large = runEval(serial, large.pos, large.z, large.box);
  auto ref_large_reuse = runEval(serial, large_reuse, large.z, large.box);
  auto ref_large_rebuild = runEval(serial, large_rebuild, large.z, large.box);

  for (const char *th : {"2", "4", "8"}) {
    INFO("large OMP_NUM_THREADS=" << th);
    ThreadGuard::set(th);
    rgpot::EMTPot pot;
    CHECK(countBitDiffs(runEval(pot, large.pos, large.z, large.box),
                        ref_large) == 0);
    const auto before = rgpot::teamsOpened();
    CHECK(countBitDiffs(runEval(pot, large_reuse, large.z, large.box),
                        ref_large_reuse) == 0);
#if defined(_OPENMP)
    CHECK(rgpot::teamsOpened() == before + 1);
#endif
    CHECK(countBitDiffs(runEval(pot, large_rebuild, large.z, large.box),
                        ref_large_rebuild) == 0);
  }
}
