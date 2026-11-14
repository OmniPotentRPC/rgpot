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
#include <unordered_map>
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

// The pair fixtures: each directed edge carries half of
// 0.5 * k * (d - r0)^2, so every pair inside the cutoff counts once.
Eval springs(const std::vector<double> &pos, double cutoff) {
  const double k = 0.75;
  const double r0 = 1.125;
  const size_t n = pos.size() / 3;
  Eval out;
  out.force.assign(pos.size(), 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      double v[3];
      for (size_t c = 0; c < 3; ++c)
        v[c] = pos[3 * i + c] - pos[3 * j + c];
      const double d = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (d >= cutoff)
        continue;
      const double s = d - r0;
      out.energy += 0.5 * k * s * s;
      for (size_t c = 0; c < 3; ++c) {
        out.force[3 * i + c] -= k * s / d * v[c];
        out.force[3 * j + c] += k * s / d * v[c];
      }
    }
  }
  return out;
}

void require_same(const Eval &a, const Eval &b, double tol) {
  REQUIRE_THAT(a.energy, Catch::Matchers::WithinAbs(b.energy, tol));
  REQUIRE(a.force.size() == b.force.size());
  for (size_t i = 0; i < a.force.size(); ++i)
    REQUIRE_THAT(a.force[i], Catch::Matchers::WithinAbs(b.force[i], tol));
}

// Four atoms, every pair inside the 6 angstrom cutoff: 12 directed
// edges, the shape the fixed pair package was compiled for.
const std::vector<double> kTetra = {12.5, 12.5, 12.5, 13.6, 12.5, 12.5,
                                    12.5, 13.7, 12.5, 12.5, 12.5, 13.4};

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

TEST_CASE("a molecular box rejects a periodic image", "[uma][graph]") {
  // After recentering, these three atoms span more than the 6 angstrom
  // cutoff and less than the 15 angstrom box, so the missing pair comes
  // back as a periodic image.
  rgpot::UmaPot pot = open_pot(package_path("UMA_MOLECULAR_BOX_PT2"));
  const std::vector<double> pos = {0, 0, 0, 0, 0, 1, 0, 0, 10};
  const std::vector<int> z = {6, 7, 1};
  const double box[9] = {15, 0, 0, 0, 15, 0, 0, 0, 15};
  std::vector<double> force(9, 0.0);
  rgpot::ForceOut slot{force.data(), 0.0, 0.0};
  const rgpot::ForceInput in{3, pos.data(), z.data(), box};
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in, &slot),
      Catch::Matchers::ContainsSubstring("cell_offsets") &&
          Catch::Matchers::ContainsSubstring("nonzero"));
}

TEST_CASE("a fixed-shape package loads and computes a force", "[uma][graph]") {
  // The keyless tetrazine package records edge_index [2, 56] in shapes
  // and neither nedges_min nor nedges_max. Eight atoms at 0.5 angstrom
  // stay inside the 10 angstrom cutoff, so the list is the traced 56.
  rgpot::UmaPot pot = open_pot(package_path("UMA_TETRAZINE_KEYLESS_PT2"));
  const Eval out = run(pot, line(8, 0.5));
  REQUIRE(std::isfinite(out.energy));
  REQUIRE_THAT(out.energy, Catch::Matchers::WithinAbs(0.0, 1e-6));
  REQUIRE(out.force.size() == 24);
  for (double component : out.force)
    REQUIRE_THAT(component, Catch::Matchers::WithinAbs(-7.0, 1e-6));
}

TEST_CASE("a dynamic package still loads", "[uma][graph]") {
  rgpot::UmaPot pot = open_pot(package_path("UMA_GATHER_DYNAMIC_PT2"));
  const Eval out = run(pot, line(4, 1.0));
  REQUIRE(std::isfinite(out.energy));
  REQUIRE_THAT(out.energy, Catch::Matchers::WithinAbs(468.0, 1e-6));
}

TEST_CASE("tetrazine past the 10 angstrom cutoff names the fixed edge shape",
          "[uma][graph]") {
  // Control packages omit nedges_min and nedges_max. shapes records
  // the traced complete pair list, [2, 56], cutoff 10 angstrom. Moving
  // one hydrogen by 12 angstroms crosses that cutoff. The named error
  // is raised before the compiled graph runs.
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
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 56]"));
}

TEST_CASE("a fixed-shape package with no shape metadata runs at its "
          "compiled shapes",
          "[uma][graph][compiled]") {
  // pair-fixed.pt2 records no nedges_min, nedges_max, natoms,
  // pos_shape, edge_index_shape or shapes. Its graph was compiled for
  // pos [4, 3] and edge_index [2, 12].
  rgpot::UmaPot fixed = open_pot(package_path("UMA_PAIR_FIXED_PT2"));
  rgpot::UmaPot dynamic = open_pot(package_path("UMA_PAIR_DYNAMIC_PT2"));
  const Eval a = run(fixed, kTetra);
  const Eval b = run(dynamic, kTetra);
  REQUIRE(std::isfinite(a.energy));
  REQUIRE(a.energy > 0.01);
  require_same(a, springs(kTetra, 6.0), 1e-10);
  require_same(a, b, 1e-12);

  std::vector<double> moved = kTetra;
  moved[4] += 0.05;
  moved[11] -= 0.08;
  require_same(run(fixed, moved), run(dynamic, moved), 1e-12);
  require_same(run(fixed, moved), springs(moved, 6.0), 1e-10);
}

TEST_CASE("a fixed-shape package names the compiled edge_index",
          "[uma][graph][compiled]") {
  // The last atom leaves the cutoff, so the list has 6 edges instead
  // of the 12 the graph was compiled for.
  rgpot::UmaPot fixed = open_pot(package_path("UMA_PAIR_FIXED_PT2"));
  std::vector<double> stretched = kTetra;
  stretched[11] = 30.0;
  try {
    run(fixed, stretched);
    FAIL("the fixed-shape package returned a force");
  } catch (const rgpot::UmaShapeError &err) {
    REQUIRE(err.field() == "edge_index");
    REQUIRE(err.actual() == "[2, 6]");
    REQUIRE(err.package() == "[2, 12]");
    REQUIRE_THAT(std::string(err.what()),
                 Catch::Matchers::Equals("UmaPot: edge_index shape is [2, 6], "
                                         "the package was exported for "
                                         "[2, 12]"));
  }

  // The dynamic package of the same model takes that list.
  rgpot::UmaPot dynamic = open_pot(package_path("UMA_PAIR_DYNAMIC_PT2"));
  require_same(run(dynamic, stretched), springs(stretched, 6.0), 1e-10);
}

TEST_CASE("a fixed-shape package names the compiled atom count",
          "[uma][graph][compiled]") {
  // Three atoms give six edges. The atom count is named first, since
  // that is the shape that does not fit.
  rgpot::UmaPot fixed = open_pot(package_path("UMA_PAIR_FIXED_PT2"));
  const std::vector<double> three(kTetra.begin(), kTetra.begin() + 9);
  try {
    run(fixed, three);
    FAIL("the fixed-shape package returned a force");
  } catch (const rgpot::UmaShapeError &err) {
    REQUIRE(err.field() == "pos");
    REQUIRE_THAT(std::string(err.what()),
                 Catch::Matchers::Equals("UmaPot: pos shape is [3, 3], the "
                                         "package was exported for [4, 3]"));
  }
  rgpot::UmaPot dynamic = open_pot(package_path("UMA_PAIR_DYNAMIC_PT2"));
  require_same(run(dynamic, three), springs(three, 6.0), 1e-10);
}

TEST_CASE("a package with no edge range is refused", "[uma][graph]") {
  // No range key, no shapes, and a wrapper source without input checks:
  // nothing records the edge count the graph was compiled for.
  rgpot::UmaPot pot = open_pot(package_path("UMA_NO_EDGES_PT2"));
  REQUIRE_THROWS_WITH(run(pot, line(3, 1.0)),
                      Catch::Matchers::ContainsSubstring("nedges") &&
                          Catch::Matchers::ContainsSubstring("absent"));
  // The second call is refused the same way.
  REQUIRE_THROWS_WITH(run(pot, line(3, 1.0)),
                      Catch::Matchers::ContainsSubstring("nedges") &&
                          Catch::Matchers::ContainsSubstring("absent"));
}

TEST_CASE("tetrazine walked past 10 angstroms names the edge count",
          "[uma][graph]") {
  // s-tetrazine, eight atoms. The static package was traced at 56
  // edges. Moving one hydrogen by 12 angstroms changes that count.
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
  try {
    pot.forceImpl(in, &slot);
    FAIL("the static package returned a force");
  } catch (const rgpot::UmaShapeError &err) {
    const std::string message = err.what();
    REQUIRE(err.field() == "edge_index");
    REQUIRE_THAT(message,
                 Catch::Matchers::ContainsSubstring("edge_index shape is"));
    REQUIRE_THAT(message, Catch::Matchers::ContainsSubstring("[2, 56]"));
    REQUIRE_THAT(message, !Catch::Matchers::ContainsSubstring("returned 2"));
  }
}

TEST_CASE("a band checks each system's edge count", "[uma][graph]") {
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
  REQUIRE_THROWS_WITH(
      pot.forceBatchImpl(batch),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 2]") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
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

TEST_CASE("a molecular band refuses a changed edge count", "[uma][graph]") {
  // batch_max > 1 used to enter the band path before the n*(n-1)
  // check, and a range package skipped the per-system check. One
  // system of this pair has left the cutoff. The named error is
  // raised from forceImpl, which is the call that used to miss it.
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
  REQUIRE_THROWS_WITH(
      pot.forceImpl(in[1], &out[1]),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
  const rgpot::ForceBatch batch{2, in, out};
  REQUIRE_THROWS_WITH(
      pot.forceBatchImpl(batch),
      Catch::Matchers::ContainsSubstring("edge_index shape is") &&
          Catch::Matchers::ContainsSubstring("[2, 2]") &&
          Catch::Matchers::ContainsSubstring("[2, 6]"));
}
