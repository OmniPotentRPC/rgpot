// MIT License
// Copyright 2023--present rgpot developers
//
// One neighbour list, and one potential instance, rebuilt across systems
// with no pair inside the cutoff: an empty system, a single atom, and atoms
// farther apart than the cutoff. Each is met as the first call, after a
// call with pairs, and before one, on open and periodic cells. A system
// with no pair has exactly zero forces, and exactly zero energy unless the
// potential gives a free atom an energy of its own (EMT counts from the
// bulk crystal, and an embedding function can be nonzero at zero density);
// then the energy is that of as many free atoms, the value the bonded
// energy reaches at the cutoff. A system with pairs gives the same numbers
// as a fresh instance.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

#include "rgpot/EAM/EAMPot.hpp"
#include "rgpot/EMT/EMTPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/StillingerWeber/SWSi.hpp"
#include "rgpot/SuttonChen/SuttonChenPot.hpp"
#include "rgpot/Tersoff/TersoffSiC.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/nlist/rgpot_nlist.h"
#include "rgpot/types/AtomMatrix.hpp"
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

using rgpot::types::AtomMatrix;

namespace {

constexpr double kBox = 120.0;

const double kBox9[9] = {kBox, 0.0, 0.0, 0.0, kBox, 0.0, 0.0, 0.0, kBox};

std::array<std::array<double, 3>, 3> cube() {
  return {{{kBox, 0.0, 0.0}, {0.0, kBox, 0.0}, {0.0, 0.0, kBox}}};
}

struct Geometry {
  std::string name;
  std::vector<double> xyz;
};

// Fifty angstroms between atoms and seventy between periodic images: past
// every default cutoff below, Sutton-Chen's 40 included.
std::vector<Geometry> zeroPairGeometries() {
  return {
      {"empty", {}},
      {"one atom", {20.0, 20.0, 20.0}},
      {"apart", {20.0, 20.0, 20.0, 70.0, 20.0, 20.0, 20.0, 70.0, 20.0}},
  };
}

// Three atoms, two bonds of length `bond` at a right angle.
Geometry bonded(double bond) {
  return {"bonded", {20.0, 20.0, 20.0, 20.0 + bond, 20.0, 20.0, 20.0, 20.0 + bond, 20.0}};
}

AtomMatrix matrixOf(const Geometry &g) {
  const std::size_t n = g.xyz.size() / 3;
  AtomMatrix positions = AtomMatrix::Zero(n, 3);
  for (std::size_t a = 0; a < n; ++a) {
    for (std::size_t d = 0; d < 3; ++d) {
      positions(a, d) = g.xyz[3 * a + d];
    }
  }
  return positions;
}

struct Result {
  double energy;
  AtomMatrix forces;
};

Result evaluate(rgpot::PotentialBase &pot, const Geometry &g, int z) {
  const auto positions = matrixOf(g);
  const std::vector<int> types(positions.rows(), z);
  auto [energy, forces, variance] = pot(positions, types, cube());
  (void)variance;
  return {energy, std::move(forces)};
}

void requireZero(const Result &r, const Geometry &g, double freeAtom) {
  INFO("zero-pair geometry " << g.name);
  REQUIRE(r.forces.rows() == g.xyz.size() / 3);
  double energy = 0.0;
  for (std::size_t a = 0; a < r.forces.rows(); ++a) {
    energy += freeAtom;
  }
  REQUIRE(r.energy == energy);
  for (std::size_t a = 0; a < r.forces.rows(); ++a) {
    for (std::size_t d = 0; d < 3; ++d) {
      REQUIRE(r.forces(a, d) == 0.0);
    }
  }
}

void requireSame(const Result &got, const Result &want) {
  REQUIRE(got.energy != 0.0);
  REQUIRE(got.energy == want.energy);
  REQUIRE(got.forces.rows() == want.forces.rows());
  for (std::size_t a = 0; a < got.forces.rows(); ++a) {
    for (std::size_t d = 0; d < 3; ++d) {
      REQUIRE(got.forces(a, d) == want.forces(a, d));
    }
  }
}

struct Case {
  const char *name;
  std::function<std::unique_ptr<rgpot::PotentialBase>()> make;
  int z;
  double bond;
  bool freeAtomEnergy = false;
};

template <typename Pot, typename... Args> Case potCase(const char *name, int z, double bond,
                                                       Args... args) {
  return {name, [=] { return std::unique_ptr<rgpot::PotentialBase>(new Pot(args...)); }, z,
          bond};
}

Case withFreeAtomEnergy(Case c) {
  c.freeAtomEnergy = true;
  return c;
}

std::vector<Case> cases() {
  std::vector<Case> all{
      potCase<rgpot::LJPot>("LJ", 1, 1.12),
      potCase<rgpot::LJClusterPot>("LJCluster", 1, 1.12),
      potCase<rgpot::MorsePot>("Morse", 78, 2.9),
      potCase<rgpot::ZBLPot>("ZBL", 14, 1.5),
      potCase<rgpot::SuttonChenPot>("SuttonChen open", 78, 1.0, rgpot::SuttonChenConfig{}),
      potCase<rgpot::SuttonChenPot>("SuttonChen periodic", 78, 1.0,
                                    rgpot::SuttonChenConfig{.periodic = true}),
      withFreeAtomEnergy(potCase<rgpot::EMTPot>("EMT", 29, 2.55)),
      withFreeAtomEnergy(potCase<rgpot::EAMPot>("EAMCell", 13, 2.86)),
      potCase<rgpot::StillingerWeberSi>("SWSi1985", 14, 2.35),
      potCase<rgpot::TersoffSiC>("TersoffSiC", 14, 2.35),
  };
#ifdef RGPOT_HAS_FORTRAN_POTS
  all.push_back(potCase<rgpot::fortranpots::SWPot>("Fortran SW", 14, 2.35));
  all.push_back(potCase<rgpot::fortranpots::TersoffPot>("Fortran Tersoff", 14, 2.35));
  all.push_back(potCase<rgpot::fortranpots::EDIPPot>("Fortran EDIP", 14, 2.35));
  all.push_back(
      withFreeAtomEnergy(potCase<rgpot::fortranpots::LenoskyPot>("Fortran Lenosky", 14, 2.35)));
  all.push_back(potCase<rgpot::fortranpots::EAMAlPot>("Fortran EAM Al", 13, 2.86));
  all.push_back(potCase<rgpot::fortranpots::FeHePot>("Fortran FeHe", 26, 2.48));
  all.push_back(potCase<rgpot::fortranpots::CuH2Pot>("Fortran CuH2", 29, 2.55));
#endif
  return all;
}

} // namespace

TEST_CASE("potentials rebuilt between zero pairs and pairs", "[nlist][zero-pairs]") {
  const auto zero = zeroPairGeometries();
  for (const auto &c : cases()) {
    DYNAMIC_SECTION(c.name) {
      const auto pairs = bonded(c.bond);
      auto fresh = c.make();
      const Result want = evaluate(*fresh, pairs, c.z);
      REQUIRE(want.energy != 0.0);
      double freeAtom = 0.0;
      if (c.freeAtomEnergy) {
        auto single = c.make();
        freeAtom = evaluate(*single, zeroPairGeometries()[1], c.z).energy;
        REQUIRE(freeAtom != 0.0);
      }

      SECTION("first call with zero pairs") {
        for (const auto &g : zero) {
          auto pot = c.make();
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
        }
      }
      SECTION("pairs, then reset to zero pairs") {
        for (const auto &g : zero) {
          auto pot = c.make();
          requireSame(evaluate(*pot, pairs, c.z), want);
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
        }
      }
      SECTION("zero pairs, then pairs again") {
        for (const auto &g : zero) {
          auto pot = c.make();
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
          requireSame(evaluate(*pot, pairs, c.z), want);
          requireZero(evaluate(*pot, g, c.z), g, freeAtom);
          requireSame(evaluate(*pot, pairs, c.z), want);
        }
      }
    }
  }
}

namespace {

std::size_t atomsOf(const Geometry &g) { return g.xyz.size() / 3; }

const double *pointsOf(const Geometry &g) {
  return g.xyz.empty() ? nullptr : g.xyz.data();
}

rgpot::nlist::Query queryFor(bool full, bool periodic, double skin) {
  rgpot::nlist::Query query;
  query.cutoff = 2.0;
  query.skin = skin;
  query.cut = skin > 0.0 ? rgpot::nlist::Cut::Loose : rgpot::nlist::Cut::Strict;
  query.shape = rgpot::nlist::Pairs{full ? rgpot::nlist::Shell::Full : rgpot::nlist::Shell::Half};
  query.periodic = {periodic, periodic, periodic};
  return query;
}

void requireTable(const rgpot::nlist::Table &table, const Geometry &g, std::size_t expected) {
  INFO("geometry " << g.name);
  REQUIRE(table.size() == expected);
  const auto view = table.view();
  REQUIRE(view.first.size() == expected);
  REQUIRE(view.second.size() == expected);
  REQUIRE(view.shift.size() == 3 * expected);
  REQUIRE(view.vector.size() == 3 * expected);
  REQUIRE(view.distance.size() == expected);
  for (std::size_t p = 0; p < expected; ++p) {
    REQUIRE(static_cast<std::size_t>(view.first[p]) < atomsOf(g));
    REQUIRE(static_cast<std::size_t>(view.second[p]) < atomsOf(g));
    REQUIRE(view.distance[p] == 1.0);
  }
}

} // namespace

TEST_CASE("the neighbour layer rebuilt between zero pairs and pairs", "[nlist][zero-pairs]") {
  const bool full = GENERATE(false, true);
  const bool periodic = GENERATE(false, true);
  const double skin = GENERATE(0.0, 0.5);
  CAPTURE(full, periodic, skin);
  const auto zero = zeroPairGeometries();
  const Geometry pairs{"pair", {20.0, 20.0, 20.0, 21.0, 20.0, 20.0}};
  const std::size_t stored = full ? 2 : 1;
  const auto query = queryFor(full, periodic, skin);

  SECTION("build") {
    for (const auto &g : zero) {
      rgpot::nlist::Table table;
      rgpot::nlist::build(table, pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(table, g, 0);
      rgpot::nlist::build(table, pointsOf(pairs), atomsOf(pairs), kBox9, query);
      requireTable(table, pairs, stored);
      rgpot::nlist::build(table, pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(table, g, 0);
      rgpot::nlist::buildByAtom(table, pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(table, g, 0);
    }
  }

  SECTION("builder") {
    for (const auto &g : zero) {
      rgpot::nlist::Builder builder;
      builder.update(pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(builder.table(), g, 0);
      builder.update(pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(builder.table(), g, 0);
      builder.update(pointsOf(pairs), atomsOf(pairs), kBox9, query);
      requireTable(builder.table(), pairs, stored);
      builder.update(pointsOf(g), atomsOf(g), kBox9, query);
      requireTable(builder.table(), g, 0);
      builder.update(pointsOf(pairs), atomsOf(pairs), kBox9, query);
      requireTable(builder.table(), pairs, stored);
    }
  }

  SECTION("walk") {
    for (const auto &g : zero) {
      rgpot::nlist::Walk walk(pointsOf(g), atomsOf(g), kBox9, query);
      std::vector<std::int32_t> first(4), second(4), shift(12);
      std::vector<double> vector(12), distance(4);
      bool done = false;
      const std::size_t got =
          walk.pull({first, second, shift, vector, distance, {}}, done);
      REQUIRE(got == 0);
    }
  }

  SECTION("C ABI") {
    rgpot_nlist_request request;
    rgpot_nlist_request_init(&request);
    request.cutoff = query.cutoff;
    request.skin = skin;
    request.cut = skin > 0.0 ? 1 : 0;
    request.shell = full ? 1 : 0;
    for (int k = 0; k < 3; ++k) {
      request.periodic[k] = periodic ? 1 : 0;
    }
    rgpot_nlist_builder *builder = nullptr;
    REQUIRE(rgpot_nlist_builder_open(&builder) == RGPOT_NLIST_OK);
    rgpot_nlist_fortran_list *list = rgpot_nlist_fortran_open();
    REQUIRE(list != nullptr);
    const std::int32_t pbc[3] = {periodic ? 1 : 0, periodic ? 1 : 0, periodic ? 1 : 0};
    auto check = [&](const Geometry &g, std::size_t want) {
      INFO("geometry " << g.name);
      std::vector<std::int32_t> first(4, -1), second(4, -1), shift(12, -1);
      std::vector<double> vector(12, -1.0), distance(4, -1.0);
      rgpot_nlist_soa soa{};
      soa.bytes = sizeof(soa);
      soa.first = first.data();
      soa.second = second.data();
      soa.shift = shift.data();
      soa.vector = vector.data();
      soa.distance = distance.data();
      soa.cap = 4;
      std::size_t count = 99;
      std::size_t needed = 99;
      REQUIRE(rgpot_nlist_build(pointsOf(g), atomsOf(g), kBox9, &request, &soa, &count,
                                &needed) == RGPOT_NLIST_OK);
      REQUIRE(count == want);
      REQUIRE(needed == want);
      std::int32_t reused = -1;
      REQUIRE(rgpot_nlist_builder_update(builder, pointsOf(g), atomsOf(g), kBox9, &request,
                                         &soa, &count, &needed, &reused) == RGPOT_NLIST_OK);
      REQUIRE(count == want);
      // The Fortran entry keeps a full list and one pair per direction.
      const std::size_t full_want = want == 0 ? 0 : 2;
      REQUIRE(rgpot_nlist_fortran_build(list, pointsOf(g), atomsOf(g), kBox9, 2.0, skin, pbc,
                                        nullptr, nullptr, nullptr, nullptr, 0, &count,
                                        &reused) == RGPOT_NLIST_OK);
      REQUIRE(count == full_want);
      REQUIRE(rgpot_nlist_fortran_build(list, pointsOf(g), atomsOf(g), kBox9, 2.0, skin, pbc,
                                        first.data(), second.data(), vector.data(),
                                        distance.data(), 4, &count,
                                        &reused) == RGPOT_NLIST_OK);
      REQUIRE(count == full_want);
      if (want == 0) {
        REQUIRE(first[0] == -1);
        REQUIRE(distance[0] == -1.0);
      }
    };
    for (const auto &g : zero) {
      check(g, 0);
      check(pairs, stored);
      check(g, 0);
      check(pairs, stored);
    }
    rgpot_nlist_fortran_close(list);
    rgpot_nlist_builder_close(builder);
  }
}
