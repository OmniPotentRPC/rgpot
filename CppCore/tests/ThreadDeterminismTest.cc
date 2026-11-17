// MIT License
// Copyright 2023--present rgpot developers

// The Tersoff, Stillinger-Weber and EAM kernels give the same energy and
// forces, bit for bit, at every thread count, on displaced cells.
//
// Each thread owns a contiguous range of atoms and the partial sums are
// added in a fixed order, so the thread count must not change a bit. Every
// comparison is between thread counts in this process. No energy or force
// is compared with a stored number, because those depend on the compiler
// and the maths library.
//
// The counts are 2, 4 and 8 against one. Where the OpenMP runtime grants a
// smaller team, the size it grants stands in for the request. With no team of
// two or more, the cases are skipped.

#include <array>
#include <cstddef>
#include <cstring>
#include <vector>

#include <catch2/catch_all.hpp>

#include "displaced_cell.hpp"

using namespace displaced;

namespace {

bool isEam(Model model) {
  return model == Model::EamStored || model == Model::EamStream;
}

std::vector<int> grantedCounts() {
  const std::vector<int> counts = Threads::multiThreadCounts();
  if (counts != std::vector<int>{2, 4, 8}) {
    WARN("the OpenMP runtime does not grant every one of 2, 4 and 8 threads");
  }
  return counts;
}

struct Variant {
  Model model = Model::TersoffSi;
  bool streamed = false;
};

const std::array<Variant, 8> kVariants = {{
    {Model::TersoffSi, false},
    {Model::TersoffSi, true},
    {Model::TersoffSiC, false},
    {Model::TersoffSiC, true},
    {Model::StillingerWeber, false},
    {Model::StillingerWeber, true},
    {Model::EamStored, false},
    {Model::EamStream, true},
}};

// Kernels change strategy with the atom count, or with the atoms per
// thread. These sizes sit on both sides of each switch: Tersoff and SW
// stored take a different walk from 4096 and from 256 atoms, SW streamed
// switches at 1024 atoms, and stored EAM takes a different pass once each
// thread owns 64 atoms.
struct Switch {
  Model model = Model::TersoffSi;
  bool streamed = false;
  std::vector<int> sizes;
};

const std::array<Switch, 5> kSwitches = {{
    {Model::TersoffSi, false, {4095, 4096}},
    {Model::TersoffSiC, false, {4095, 4096}},
    {Model::StillingerWeber, false, {255, 256}},
    {Model::StillingerWeber, true, {1023, 1024}},
    {Model::EamStored, false, {255, 256, 511, 512}},
}};

} // namespace

TEST_CASE("the displaced cell repeats and has the stated spread",
          "[displaced]") {
  for (int n : {64, 256, 4096}) {
    CAPTURE(n);
    const Cell a = fcc(n);
    const Cell b = fcc(n);
    REQUIRE(a.pos.size() == static_cast<std::size_t>(3 * n));
    REQUIRE(std::memcmp(a.pos.data(), b.pos.data(),
                        a.pos.size() * sizeof(double)) == 0);
    const Cell other = fcc(n, kSeed + 1);
    REQUIRE(std::memcmp(a.pos.data(), other.pos.data(),
                        a.pos.size() * sizeof(double)) != 0);
    REQUIRE(rmsDisplacement(a) == Catch::Approx(kHalfWidth).margin(0.005));
  }
}

TEST_CASE("displaced cells give the same bits at every thread count",
          "[threads][bits][displaced]") {
  const Threads threads;
  const std::vector<int> counts = grantedCounts();
  if (counts.empty()) {
    SKIP("the OpenMP runtime grants no team of two or more threads");
  }
  const Variant &var =
      kVariants[GENERATE(range(std::size_t{0}, kVariants.size()))];
  const int n = GENERATE(64, 256);
  CAPTURE(name(var.model), var.streamed, n);
  const Cell cell = fcc(n);

  threads.use(1);
  const Result serial = Engine(var.model, var.streamed)(cell.pos, cell.box);
  REQUIRE(serial.force.size() == cell.pos.size());
  for (int count : counts) {
    CAPTURE(count);
    threads.use(count);
    Engine parallel(var.model, var.streamed);
    const Result got = parallel(cell.pos, cell.box);
    REQUIRE(difference(got, serial) == "");
    if (isEam(var.model)) {
      REQUIRE(parallel.threadsUsed() == count);
    }
  }
}

TEST_CASE("displaced zincblende Si-C gives the same bits at every thread count",
          "[threads][bits][displaced]") {
  const Threads threads;
  const std::vector<int> counts = grantedCounts();
  if (counts.empty()) {
    SKIP("the OpenMP runtime grants no team of two or more threads");
  }
  const bool streamed = GENERATE(false, true);
  CAPTURE(streamed);
  const Cell cell = zincblende(64);
  const std::vector<int> z = zincblendeSpecies(64);

  threads.use(1);
  const Result serial = Engine(Model::TersoffSiC, streamed)(cell.pos, z, cell.box);
  REQUIRE(serial.force.size() == cell.pos.size());
  for (int count : counts) {
    CAPTURE(count);
    threads.use(count);
    Engine parallel(Model::TersoffSiC, streamed);
    const Result got = parallel(cell.pos, z, cell.box);
    REQUIRE(difference(got, serial) == "");
  }
}

TEST_CASE("one object keeps its bits as the thread count changes",
          "[threads][bits][displaced]") {
  const Threads threads;
  const std::vector<int> counts = grantedCounts();
  if (counts.empty()) {
    SKIP("the OpenMP runtime grants no team of two or more threads");
  }
  const Variant &var =
      kVariants[GENERATE(range(std::size_t{0}, kVariants.size()))];
  const int n = GENERATE(64, 256);
  CAPTURE(name(var.model), var.streamed, n);
  const Cell cell = fcc(n);

  threads.use(1);
  Engine engine(var.model, var.streamed);
  const Result serial = engine(cell.pos, cell.box);

  // Up through the counts, back down, and twice at the same count. Every
  // call after the first finds the list the first one built.
  std::vector<int> order = {1};
  order.insert(order.end(), counts.begin(), counts.end());
  order.insert(order.end(), counts.rbegin(), counts.rend());
  order.insert(order.end(), {1, 1});
  for (int count : order) {
    CAPTURE(count);
    threads.use(count);
    const Result got = engine(cell.pos, cell.box);
    REQUIRE(difference(got, serial) == "");
    if (isEam(var.model)) {
      REQUIRE(engine.threadsUsed() == count);
    }
  }
  if (var.model == Model::EamStored) {
    REQUIRE(engine.listBuilds() == 1);
  }
}

TEST_CASE("sizes on both sides of a path switch give the same bits",
          "[threads][bits][displaced]") {
  const Threads threads;
  const std::vector<int> counts = grantedCounts();
  if (counts.empty()) {
    SKIP("the OpenMP runtime grants no team of two or more threads");
  }
  const Switch &row =
      kSwitches[GENERATE(range(std::size_t{0}, kSwitches.size()))];

  for (int n : row.sizes) {
    CAPTURE(name(row.model), row.streamed, n);
    const Cell cell = fcc(n);
    threads.use(1);
    const Result serial = Engine(row.model, row.streamed)(cell.pos, cell.box);
    for (int count : counts) {
      CAPTURE(count);
      threads.use(count);
      Engine parallel(row.model, row.streamed);
      const Result got = parallel(cell.pos, cell.box);
      REQUIRE(difference(got, serial) == "");
      if (isEam(row.model)) {
        REQUIRE(parallel.threadsUsed() == count);
      }
    }
  }
}
