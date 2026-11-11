// Copyright 2023--present rgpot developers
//
// A float64 package must resolve an energy difference of about 1e-7
// Angstrom on a three-atom line. Coordinates are exact float32 values
// (12.5, 11.5 and 13.5), so casting the positions to float32 makes the
// shifted geometry identical to the original and the energy difference
// is 0. Meson builds the packages and passes their paths in.

#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaEdges.hpp"
#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

std::string package_path(const char *name) {
  const char *path = std::getenv(name);
  if (path == nullptr || path[0] == '\0') {
    FAIL(std::string(name) + " is not set");
  }
  return path;
}

struct Eval {
  double energy = 0.0;
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

void require_six_pairs(const std::vector<double> &pos) {
  rgpot::UmaEdgeSession session;
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "6"},
      {"nedges_max", "6"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  const rgpot::ForceInput in{3, pos.data(), z.data(), box};
  const auto &edges = rgpot::build_system_edges(
      in, 6.0, 6.0, 0.0, 300, contract, 1, session);
  REQUIRE(edges.neighbor.size() == 6);
  REQUIRE(edges.center.size() == 6);
  for (float shift : edges.offsets)
    REQUIRE(shift == 0.f);
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
  rgpot::UmaPot pot = open_pot(package_path(env));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  const double difference = second.energy - first.energy;
  REQUIRE_THAT(first.energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE(std::fabs(difference) > 1e-9);
  REQUIRE_THAT(difference, Catch::Matchers::WithinAbs(1e-7, 1e-9));
}

} // namespace

TEST_CASE("float64 package resolves a 1e-7 angstrom shift", "[uma][float64]") {
  expect_resolved("UMA_FLOAT64_PT2");
}

TEST_CASE("model_dtype alone resolves a 1e-7 angstrom shift",
          "[uma][float64]") {
  expect_resolved("UMA_FLOAT64_MODEL_ONLY_PT2");
}

TEST_CASE("float32 package collapses a 1e-7 angstrom shift",
          "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT32_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  REQUIRE_THAT(first.energy, Catch::Matchers::WithinAbs(112.5, 1e-4));
  REQUIRE(second.energy - first.energy == 0.0);
}

TEST_CASE("a package with neither dtype key stays float32", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT32_NOMETA_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  REQUIRE(second.energy - first.energy == 0.0);
}

TEST_CASE("disagreeing dtype keys are refused", "[uma][float64]") {
  rgpot::UmaConfig cfg;
  cfg.model_path = package_path("UMA_DTYPE_MISMATCH_PT2");
  cfg.device = "cpu";
  cfg.task_name = "omol";
  cfg.charge = 0;
  cfg.spin = 1;
  rgpot::UmaPot pot(cfg);
  const std::vector<double> base = line_geometry();
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> force(base.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{base.size() / 3, base.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("pos_dtype is float32") &&
          Catch::Matchers::ContainsSubstring("model_dtype is float64"));
}

TEST_CASE("finite difference step of 1e-6 matches the sum", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_PT2"));
  const std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  const double step = 1e-6;
  shifted.back() += step;
  const Eval first = run(pot, base);
  const Eval second = run(pot, shifted);
  const double slope = (second.energy - first.energy) / step;
  REQUIRE_THAT(slope, Catch::Matchers::WithinAbs(1.0, 1e-6));
}

TEST_CASE("a complete float64 list keeps the moved atom",
          "[uma][float64]") {
  // The three-atom line was traced at six edges. Moving one atom
  // past the old cutoff keeps that count. The energy is the sum of
  // the coordinates, so the moved atom is still in the result.
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_PT2"));
  std::vector<double> pos = line_geometry();
  pos.back() = 30.0;
  const Eval got = run(pot, pos);
  REQUIRE_THAT(got.energy, Catch::Matchers::WithinAbs(131.0, 1e-9));
  require_six_pairs(pos);
}

TEST_CASE("batch path resolves a 1e-7 angstrom shift", "[uma][float64]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_FLOAT64_BATCH_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  std::vector<double> base = line_geometry();
  std::vector<double> shifted = base;
  shifted.back() += 1e-7;
  std::vector<double> force0(9, 0.0), force1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, base.data(), z.data(), box},
      {3, shifted.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {force0.data(), 0.0, 0.0},
      {force1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatchImpl(batch);
  REQUIRE_THAT(out[0].energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE_THAT(out[1].energy - out[0].energy,
               Catch::Matchers::WithinAbs(1e-7, 1e-9));
}
