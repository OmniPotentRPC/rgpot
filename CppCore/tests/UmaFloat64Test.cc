// Copyright (c) rgpot developers
//
// A float64 package must resolve an energy difference of about 1e-7
// Angstrom on a three-atom line. Coordinates are exact float32 values
// (12.5, 11.5, 13.5), so casting the positions to float32 makes the
// shifted geometry identical to the original and dE is 0.

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

std::string package_path(const char *name) {
  const char *path = std::getenv(name);
  if (path == nullptr || path[0] == '\0') {
    return {};
  }
  return path;
}

struct Eval {
  double energy;
  std::vector<double> force;
};

Eval run(rgpot::UmaPot &pot, std::vector<double> pos) {
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  Eval out;
  out.force.assign(pos.size(), 0.0);
  rgpot::ForceOut slot{out.force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  out.energy = slot.energy;
  return out;
}

std::vector<double> line_geometry() {
  return {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5, 11.5};
}

rgpot::UmaPot open_pot(const std::string &path) {
  rgpot::UmaConfig cfg;
  cfg.model_path = path;
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  return rgpot::UmaPot(cfg);
}

void expect_resolved(const char *env) {
  const std::string path = package_path(env);
  if (path.empty()) {
    GTEST_SKIP() << env << " is not set";
  }
  rgpot::UmaPot pot = open_pot(path);
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval a = run(pot, base);
  const Eval b = run(pot, shifted);
  const double dE = b.energy - a.energy;
  EXPECT_NEAR(a.energy, 112.5, 1e-9);
  EXPECT_GT(std::fabs(dE), 1e-9);
  EXPECT_NEAR(dE, 1e-7, 1e-9);
}

} // namespace

TEST(UmaFloat64, both_dtype_keys_resolve_a_1e_7_shift) {
  expect_resolved("UMA_FLOAT64_PT2");
}

TEST(UmaFloat64, model_dtype_alone_resolves_a_1e_7_shift) {
  expect_resolved("UMA_FLOAT64_MODEL_ONLY_PT2");
}

TEST(UmaFloat64, float32_package_collapses_a_1e_7_shift) {
  const std::string path = package_path("UMA_FLOAT32_PT2");
  if (path.empty()) {
    GTEST_SKIP() << "UMA_FLOAT32_PT2 is not set";
  }
  rgpot::UmaPot pot = open_pot(path);
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval a = run(pot, base);
  const Eval b = run(pot, shifted);
  EXPECT_NEAR(a.energy, 112.5, 1e-4);
  EXPECT_EQ(b.energy - a.energy, 0.0);
}

TEST(UmaFloat64, missing_dtype_metadata_stays_float32) {
  const std::string path = package_path("UMA_FLOAT64_NOMETA_PT2");
  if (path.empty()) {
    GTEST_SKIP() << "UMA_FLOAT64_NOMETA_PT2 is not set";
  }
  rgpot::UmaPot pot = open_pot(path);
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval a = run(pot, base);
  const Eval b = run(pot, shifted);
  EXPECT_EQ(b.energy - a.energy, 0.0);
}

TEST(UmaFloat64, finite_difference_step_of_1e_6_matches_the_sum) {
  const std::string path = package_path("UMA_FLOAT64_PT2");
  if (path.empty()) {
    GTEST_SKIP() << "UMA_FLOAT64_PT2 is not set";
  }
  rgpot::UmaPot pot = open_pot(path);
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  const double step = 1e-6;
  shifted.back() += step;
  const Eval a = run(pot, base);
  const Eval b = run(pot, shifted);
  const double slope = (b.energy - a.energy) / step;
  EXPECT_NEAR(slope, 1.0, 1e-6);
}

TEST(UmaFloat64, batch_path_resolves_a_1e_7_shift) {
  const std::string path = package_path("UMA_FLOAT64_BATCH_PT2");
  if (path.empty()) {
    GTEST_SKIP() << "UMA_FLOAT64_BATCH_PT2 is not set";
  }
  rgpot::UmaPot pot = open_pot(path);
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  std::vector<double> f0(9, 0.0), f1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, base.data(), z.data(), box},
      {3, shifted.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {f0.data(), 0.0, 0.0},
      {f1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatchImpl(batch);
  EXPECT_NEAR(out[0].energy, 112.5, 1e-9);
  EXPECT_NEAR(out[1].energy - out[0].energy, 1e-7, 1e-9);
}
