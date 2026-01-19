// MIT License
// Copyright 2026--present Rohit Goswami <HaoZeke>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <random>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PotentialCache.hpp"

#include <rocksdb/db.h>
#include <rocksdb/options.h>

using namespace Catch::Matchers;
using namespace std::chrono;

TEST_CASE("Potential caching with rgpot", "[Potential]") {
  // Setup RocksDB
  rocksdb::DB *db_ptr;
  rocksdb::Options options;
  options.create_if_missing = true;
  std::string db_path = "/tmp/rgpot_test_rocksdb";
  rocksdb::DestroyDB(db_path, options);
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_ptr);
  REQUIRE(status.ok());
  auto db = std::unique_ptr<rocksdb::DB, void (*)(rocksdb::DB *)>(
      db_ptr, [](rocksdb::DB *ptr) { delete ptr; });
  // Setup System
  const int n_atoms = 128;
  rgpot::types::AtomMatrix positions(n_atoms, 3);
  std::mt19937 gen(1644009449);
  std::uniform_real_distribution<> dis(
      0.0, 20.0); // Increase volume to lower density
  for (size_t i = 0; i < n_atoms * 3; ++i) {
    positions.data()[i] = dis(gen);
  }

  std::vector<int> types(n_atoms, 1);
  std::array<std::array<double, 3>, 3> box = {
      {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}};

  // Initialize Potential and Cache Wrapper
  auto pot = std::make_shared<rgpot::LJPot>();
  auto pcache = rgpot::cache::PotentialCache();
  pcache.set_cache(db.get());
  pot->set_cache(&pcache);

  // Initial cache miss
  auto start = high_resolution_clock::now();
  auto [energy1, f1] = (*pot)(positions, types, box);
  auto end = high_resolution_clock::now();
  auto base_call = duration_cast<nanoseconds>(end - start).count();

  SECTION("Cache hit") {
    start = high_resolution_clock::now();
    auto [energy2, f2] = (*pot)(positions, types, box);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<nanoseconds>(end2 - start).count();

    REQUIRE_THAT(energy1, WithinAbs(energy2, 1e-6));
    // Cache hit should be significantly faster
    // Note: On very small systems LJ is so fast this might be noisy,
    // but logic holds.
    REQUIRE(duration2 <= base_call);
  }

  SECTION("Cache invalidation on position change") {
    // Change positions
    positions(1, 0) = 20.0;

    start = high_resolution_clock::now();
    auto [energy3, f3] = (*pot)(positions, types, box);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<nanoseconds>(end2 - start).count();

    REQUIRE(energy3 != energy1);
    // Cache miss roughly same as base
    // Allow some jitter
    REQUIRE(duration2 > (base_call / 100));
  }

  SECTION("Multiple potentials sharing cache") {
    auto pot2 = std::make_shared<rgpot::LJPot>();
    pot2->set_cache(&pcache); // Share the underlying cache

    start = high_resolution_clock::now();
    auto [energy2, f2] = (*pot2)(positions, types, box); // Same pos as initial
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<nanoseconds>(end2 - start).count();

    // Should hit cache populated by 'pot'
    REQUIRE_THAT(energy2, WithinAbs(energy1, 1e-6));
    REQUIRE(duration2 <= base_call);
  }
}
