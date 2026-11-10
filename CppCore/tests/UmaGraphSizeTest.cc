// Copyright (c) rgpot developers
//
// A universal model for atoms package compiled for one neighbour list
// must not return a force when that list changes size. The error names
// the shape that was passed and the shape the package was exported for.
// A package exported with a symbolic edge count returns finite forces
// that match the eager gather.

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/UmaPot/UmaPot.hpp"

namespace {

std::string package_path(const char *name) {
  const char *path = std::getenv(name);
  if (path == nullptr || path[0] == '\0') {
    FAIL(std::string(name) + " is not set");
  }
  return path;
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

struct Eval {
  double energy = 0.0;
  std::vector<double> force;
};

Eval run(rgpot::UmaPot &pot, const std::vector<double> &pos) {
  std::vector<int> z(pos.size() / 3, 6);
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  Eval out;
  out.force.assign(pos.size(), 0.0);
  rgpot::ForceOut slot{out.force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{pos.size() / 3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  out.energy = slot.energy;
  return out;
}

std::vector<double> line(int n, double spacing) {
  std::vector<double> pos(static_cast<size_t>(n) * 3);
  for (int i = 0; i < n; ++i) {
    pos[static_cast<size_t>(3 * i)] = 12.5;
    pos[static_cast<size_t>(3 * i + 1)] = 12.5;
    pos[static_cast<size_t>(3 * i + 2)] = 12.5 + spacing * i;
  }
  return pos;
}

} // namespace

TEST_CASE("tetrazine-sized static graph names both edge shapes",
          "[uma][graph]") {
  // Eight atoms, every pair inside the cutoff: 8 * 7 = 56 edges.
  // Moving one atom past the cutoff changes the neighbour list.
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_STATIC_PT2"));
  std::vector<double> pos = line(8, 0.5);
  pos.back() = 40.0;
  bool threw = false;
  try {
    run(pot, pos);
  } catch (const rgpot::UmaShapeError &err) {
    threw = true;
    const std::string message = err.what();
    REQUIRE(err.field() == "edge_index");
    REQUIRE_THAT(message,
                 Catch::Matchers::ContainsSubstring("edge_index shape is"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 42]"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring(
                              "the package was exported for [2, 56]"));
  }
  REQUIRE(threw);
}

TEST_CASE("dynamic gather matches eager forces when the neighbour list changes",
          "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  const Eval compact = run(pot, line(4, 1.0));
  std::vector<double> stretched = line(4, 1.0);
  stretched.back() = 40.0;
  const Eval far = run(pot, stretched);

  REQUIRE(std::isfinite(compact.energy));
  REQUIRE(std::isfinite(far.energy));
  REQUIRE_THAT(compact.energy, Catch::Matchers::WithinAbs(468.0, 1e-6));
  REQUIRE_THAT(far.energy, Catch::Matchers::WithinAbs(231.0, 1e-6));
  for (int i = 0; i < 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      REQUIRE_THAT(compact.force[static_cast<size_t>(3 * i + k)],
                   Catch::Matchers::WithinAbs(3.0, 1e-6));
    }
  }
  for (int k = 0; k < 3; ++k) {
    REQUIRE_THAT(far.force[static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(2.0, 1e-6));
    REQUIRE_THAT(far.force[9 + static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(0.0, 1e-6));
  }
}
