// MIT License
// Copyright 2023--present rgpot developers
#include <catch2/catch_all.hpp>
#include <chrono>
#include <filesystem>
#include <random>

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/PotentialCache.hpp"

#include <rocksdb/db.h>
#include <rocksdb/options.h>

using namespace Catch::Matchers;
using namespace std::chrono;
namespace fs = std::filesystem;

TEST_CASE("Potential caching with rgpot", "[Potential]") {
  // --- Common System Setup ---
  const int n_atoms = 128;
  rgpot::types::AtomMatrix positions(n_atoms, 3);
  // Use fixed seed for reproducibility
  std::mt19937 gen(1644009449);
  std::uniform_real_distribution<> dis(0.0, 20.0);
  for (size_t i = 0; i < n_atoms * 3; ++i) {
    positions.data()[i] = dis(gen);
  }

  std::vector<int> types(n_atoms, 1);
  std::array<std::array<double, 3>, 3> box = {
      {{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}};

  auto pot = std::make_shared<rgpot::LJPot>();

  // Baseline timing (no cache attached)
  auto start_base = high_resolution_clock::now();
  auto [e_base, f_base] = (*pot)(positions, types, box);
  auto end_base = high_resolution_clock::now();
  auto base_duration =
      duration_cast<nanoseconds>(end_base - start_base).count();

  SECTION("Manual DB Management (Raw Pointer)") {
    // Setup RocksDB Manually
    rocksdb::DB *db_ptr;
    rocksdb::Options options;
    options.create_if_missing = true;
    std::string db_path = "/tmp/rgpot_test_rocksdb_manual";
    rocksdb::DestroyDB(db_path, options);
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_ptr);
    REQUIRE(status.ok());

    // RAII wrapper for test safety
    auto db = std::unique_ptr<rocksdb::DB, void (*)(rocksdb::DB *)>(
        db_ptr, [](rocksdb::DB *ptr) { delete ptr; });

    auto pcache = rgpot::cache::PotentialCache();
    pcache.set_db(db.get());
    pot->set_cache(&pcache);

    // 1. Miss & Write
    auto [e1, f1] = (*pot)(positions, types, box);
    REQUIRE_THAT(e1, WithinAbs(e_base, 1e-12));

    // 2. Hit & Read
    auto start = high_resolution_clock::now();
    auto [e2, f2] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();
    auto dur = duration_cast<nanoseconds>(end - start).count();

    REQUIRE_THAT(e2, WithinAbs(e1, 1e-12));
    REQUIRE(dur < base_duration * 4);
  }

  SECTION("Managed DB Life-cycle (Path String)") {
    std::string db_path = "/tmp/rgpot_test_rocksdb_managed";
    // Ensure clean state
    rocksdb::Options opts;
    rocksdb::DestroyDB(db_path, opts);

    {
      auto pcache = rgpot::cache::PotentialCache(db_path);
      pot->set_cache(&pcache);

      // 1. Miss
      (*pot)(positions, types, box);

      // 2. Hit
      auto start = high_resolution_clock::now();
      auto [e2, f2] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      REQUIRE_THAT(e2, WithinAbs(e_base, 1e-12));
      REQUIRE(dur < base_duration * 4);
    }
    // pcache goes out of scope here, should close DB cleanly

    REQUIRE(fs::exists(db_path));
  }

  SECTION("Persistence (Close and Reopen)") {
    std::string db_path = "/tmp/rgpot_test_rocksdb_persist";
    rocksdb::Options opts;
    rocksdb::DestroyDB(db_path, opts);

    // Phase 1: Create cache, write data, destroy object
    {
      auto pcache_write = rgpot::cache::PotentialCache(db_path);
      pot->set_cache(&pcache_write);
      (*pot)(positions, types, box); // Writes to DB
    }

    // Phase 2: Create NEW cache object pointing to SAME path
    {
      auto pcache_read = rgpot::cache::PotentialCache(db_path);
      pot->set_cache(&pcache_read);

      auto start = high_resolution_clock::now();
      auto [e_read, f_read] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      // Should be a Hit (fast) despite being a new object
      REQUIRE_THAT(e_read, WithinAbs(e_base, 1e-12));
      REQUIRE(dur < base_duration * 4);
    }
  }

  SECTION("Uninitialized Cache (Graceful Degradation)") {
    // Cache object created but no DB set
    auto pcache_empty = rgpot::cache::PotentialCache();
    pot->set_cache(&pcache_empty);

    // Should call forceImpl directly without crashing
    auto start = high_resolution_clock::now();
    auto [e, f] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();

    REQUIRE_THAT(e, WithinAbs(e_base, 1e-12));
    // Should NOT be faster than base (it effectively IS base overhead)
    // We just check it didn't throw exceptions
  }
}
