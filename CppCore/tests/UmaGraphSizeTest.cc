// Copyright 2023--present rgpot developers
//
// A package traced at n(n-1) edges keeps every ordered pair when a
// contact grows, so the compiled shape does not change. A symbolic
// edge count still follows the neighbour list and matches the eager
// gather. A package with no recorded edge range is refused.

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

// The traced count is n(n-1). A truncated neighbour list fails this
// even when the energy of a sum model stays finite.
void require_ordered_pairs(const double *pos, std::size_t n, const double *box,
                           double molecular_box, int64_t recorded) {
  rgpot::UmaEdgeSession session;
  const auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", std::to_string(recorded)},
      {"nedges_max", std::to_string(recorded)},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  std::vector<int> z(n, 6);
  const rgpot::ForceInput in{n, pos, z.data(), box};
  const auto &edges = rgpot::build_system_edges(
      in, 6.0, 6.0, molecular_box, 300, contract, 1, session);
  const std::size_t want = n > 1 ? n * (n - 1) : 0;
  REQUIRE(edges.neighbor.size() == want);
  REQUIRE(edges.center.size() == want);
  REQUIRE(recorded == static_cast<int64_t>(want));
  for (float shift : edges.offsets)
    REQUIRE(shift == 0.f);
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

TEST_CASE("a complete static graph keeps every ordered pair",
          "[uma][graph]") {
  // Eight atoms were traced at 8 * 7 = 56 edges. Moving one atom
  // past the old cutoff keeps that count. Each atom is a neighbour
  // seven times, so the energy is seven times the coordinate sum.
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_STATIC_PT2"));
  std::vector<double> pos = line(8, 0.5);
  pos.back() = 40.0;
  const Eval got = run(pot, pos);
  double sum = 0.0;
  for (double value : pos)
    sum += value;
  REQUIRE_THAT(got.energy, Catch::Matchers::WithinAbs(7.0 * sum, 1e-6));
  const double wide[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  require_ordered_pairs(pos.data(), 8, wide, 0.0, 56);
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
                   Catch::Matchers::WithinAbs(-3.0, 1e-6));
    }
  }
  for (int k = 0; k < 3; ++k) {
    REQUIRE_THAT(far.force[static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(-2.0, 1e-6));
    REQUIRE_THAT(far.force[9 + static_cast<size_t>(k)],
                 Catch::Matchers::WithinAbs(0.0, 1e-6));
  }

  // A 1e-4 angstrom step does not cross the cutoff, so the compiled
  // edge count stays put. The reported force is minus dE/dx.
  std::vector<double> nudged = line(4, 1.0);
  const double step = 1e-4;
  nudged[0] += step;
  const Eval after = run(pot, nudged);
  const double finite = -(after.energy - compact.energy) / step;
  REQUIRE_THAT(finite, Catch::Matchers::WithinAbs(compact.force[0], 1e-3));
}

TEST_CASE("a molecular box keeps a long contact at shift zero",
          "[uma][graph]") {
  // The three atoms span more than a 6 angstrom cutoff and less than
  // the 15 angstrom box. The list is every ordered pair at shift 0,
  // so the call does not report a periodic image.
  rgpot::UmaPot pot = open_pot(package_path("UMA_MOLECULAR_BOX_PT2"));
  const std::vector<double> pos = {0, 0, 0, 0, 0, 1, 0, 0, 10};
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {15, 0, 0, 0, 15, 0, 0, 0, 15};
  std::vector<double> force(9, 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{3, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  REQUIRE_THAT(slot.energy, Catch::Matchers::WithinAbs(0.0, 1e-9));
  require_ordered_pairs(pos.data(), 3, box, 15.0, 6);
}

TEST_CASE("tetrazine past the 10 angstrom cutoff names the missing edge range",
          "[uma][graph]") {
  // Control packages omit nedges_min and nedges_max. They were traced
  // on the complete pair list, cutoff 10 angstrom. Moving one hydrogen
  // by 12 angstroms crosses that cutoff. The named error is raised
  // before the compiled graph runs.
  std::vector<double> pos = {
      12.5, 13.83, 12.5, 12.5, 11.17, 12.5, 13.65, 13.165, 12.5,
      13.65, 11.835, 12.5, 11.35, 13.165, 12.5, 11.35, 11.835, 12.5,
      12.5, 14.91, 12.5, 12.5, 10.09, 12.5,
  };
  pos[7 * 3 + 1] += 12.0;
  const std::vector<int> z = {6, 6, 7, 7, 7, 7, 1, 1};
  const double box[9] = {25, 0, 0, 0, 25, 0, 0, 0, 25};
  rgpot::UmaPot pot = open_pot(package_path("UMA_TETRAZINE_KEYLESS_PT2"));
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("nedges") &&
          Catch::Matchers::ContainsSubstring("absent"));
}

TEST_CASE("a package with no edge range is refused", "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_NO_EDGES_PT2"));
  REQUIRE_THROWS_WITH(run(pot, line(3, 1.0)),
                      Catch::Matchers::ContainsSubstring("nedges") &&
                          Catch::Matchers::ContainsSubstring("absent"));
}

TEST_CASE("tetrazine walked past 10 angstroms keeps 56 edges",
          "[uma][graph]") {
  // s-tetrazine, eight atoms. The static package was traced at 56
  // edges. Moving one hydrogen by 12 angstroms used to shrink the
  // vesin list. The complete count stays, and each atom is a
  // neighbour seven times.
  std::vector<double> pos = {
      12.5, 13.83, 12.5, 12.5, 11.17, 12.5, 13.65, 13.165, 12.5,
      13.65, 11.835, 12.5, 11.35, 13.165, 12.5, 11.35, 11.835, 12.5,
      12.5, 14.91, 12.5, 12.5, 10.09, 12.5,
  };
  pos[7 * 3 + 1] += 12.0;
  const std::vector<int> z = {6, 6, 7, 7, 7, 7, 1, 1};
  const double box[9] = {40, 0, 0, 0, 40, 0, 0, 0, 40};
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_STATIC_PT2"));
  std::vector<double> force(pos.size(), 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{8, pos.data(), z.data(), box};
  pot.forceImpl(in, &slot);
  double sum = 0.0;
  for (double value : pos)
    sum += value;
  REQUIRE_THAT(slot.energy, Catch::Matchers::WithinAbs(7.0 * sum, 1e-6));
  require_ordered_pairs(pos.data(), 8, box, 0.0, 56);
}

TEST_CASE("a static band keeps six edges when one system stretches",
          "[uma][graph]") {
  // Three atoms were traced at six edges. The stretched system used
  // to drop to two. Both systems stay on the complete list.
  rgpot::UmaPot pot = open_pot(package_path("UMA_BATCH_STATIC_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {80, 0, 0, 0, 80, 0, 0, 0, 80};
  std::vector<double> compact = {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5,
                                 11.5};
  std::vector<double> stretched = compact;
  stretched.back() = 40.0;
  std::vector<double> f0(9, 0.0), f1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, compact.data(), z.data(), box},
      {3, stretched.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {f0.data(), 0.0, 0.0},
      {f1.data(), 0.0, 0.0},
  };
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatchImpl(batch);
  REQUIRE_THAT(out[0].energy, Catch::Matchers::WithinAbs(112.5, 1e-9));
  REQUIRE_THAT(out[1].energy, Catch::Matchers::WithinAbs(141.0, 1e-9));
  require_ordered_pairs(compact.data(), 3, box, 0.0, 6);
  require_ordered_pairs(stretched.data(), 3, box, 0.0, 6);
}

TEST_CASE("each system of a static band is accepted", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "12"},
      {"nedges_max", "12"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t even[2] = {6, 6};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, even));
  const int64_t uneven[2] = {4, 8};
  REQUIRE_THROWS_AS(contract.checkEachSystem(2, uneven), rgpot::UmaShapeError);
}

TEST_CASE("a band range checks each system's edge count", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "1"},
      {"nedges_max", "8"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t inside[2] = {6, 2};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, inside));
  const int64_t outside[2] = {6, 9};
  REQUIRE_THROWS_AS(contract.checkEachSystem(2, outside), rgpot::UmaShapeError);
}

TEST_CASE("a molecular band requires the complete pair list", "[uma][contract]") {
  auto meta = std::unordered_map<std::string, std::string>{
      {"nedges_min", "1"},
      {"nedges_max", "100"},
  };
  const auto contract = rgpot::UmaContract::fromMetadata(meta);
  const int64_t complete[2] = {6, 6};
  REQUIRE_NOTHROW(contract.checkEachSystem(2, complete, 3, true));
  const int64_t dropped[2] = {6, 4};
  REQUIRE_THROWS_WITH(contract.checkEachSystem(2, dropped, 3, true),
                      Catch::Matchers::ContainsSubstring("[2, 4]") &&
                          Catch::Matchers::ContainsSubstring("[2, 6]"));
}

TEST_CASE("a molecular band keeps a complete pair list", "[uma][graph]") {
  // One system has left the old cutoff. The traced count is n(n-1),
  // so both the single call and the band keep six edges per system.
  rgpot::UmaPot pot = open_pot(package_path("UMA_BATCH_MOLECULAR_PT2"));
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {80, 0, 0, 0, 80, 0, 0, 0, 80};
  std::vector<double> compact = {12.5, 12.5, 12.5, 12.5, 12.5, 13.5, 12.5, 12.5,
                                 11.5};
  std::vector<double> stretched = compact;
  stretched.back() = 40.0;
  std::vector<double> f0(9, 0.0), f1(9, 0.0);
  rgpot::ForceInput in[2] = {
      {3, compact.data(), z.data(), box},
      {3, stretched.data(), z.data(), box},
  };
  rgpot::ForceOut out[2] = {
      {f0.data(), 0.0, 0.0},
      {f1.data(), 0.0, 0.0},
  };
  pot.forceImpl(in[1], &out[1]);
  REQUIRE_THAT(out[1].energy, Catch::Matchers::WithinAbs(0.0, 1e-9));
  require_ordered_pairs(stretched.data(), 3, box, 80.0, 6);
  const rgpot::ForceBatch batch{2, in, out};
  pot.forceBatchImpl(batch);
  REQUIRE_THAT(out[0].energy, Catch::Matchers::WithinAbs(0.0, 1e-9));
  REQUIRE_THAT(out[1].energy, Catch::Matchers::WithinAbs(0.0, 1e-9));
  require_ordered_pairs(compact.data(), 3, box, 80.0, 6);
}
