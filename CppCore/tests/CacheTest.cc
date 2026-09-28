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
  auto [e_base, f_base, v_base] = (*pot)(positions, types, box);
  auto end_base = high_resolution_clock::now();
  auto base_duration =
      duration_cast<nanoseconds>(end_base - start_base).count();

  SECTION("Manual DB Management (Raw Pointer)") {
    // Setup RocksDB Manually
    rocksdb::Options options;
    options.create_if_missing = true;
    std::string db_path = "/tmp/rgpot_test_rocksdb_manual";
    rocksdb::DestroyDB(db_path, options);
#if RGPOT_ROCKSDB_UNIQUE_PTR_OPEN
    std::unique_ptr<rocksdb::DB> db;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
#else
    rocksdb::DB *db_ptr = nullptr;
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_ptr);
    std::unique_ptr<rocksdb::DB> db(db_ptr);
#endif
    REQUIRE(status.ok());

    auto pcache = rgpot::cache::PotentialCache();
    pcache.set_db(db.get());
    pot->set_cache(&pcache);

    // 1. Miss & Write
    auto [e1, f1, v1] = (*pot)(positions, types, box);
    REQUIRE_THAT(e1, WithinAbs(e_base, 1e-12));

    // 2. Hit & Read
    auto start = high_resolution_clock::now();
    auto [e2, f2, v2] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();
    auto dur = duration_cast<nanoseconds>(end - start).count();

    REQUIRE_THAT(e2, WithinAbs(e1, 1e-12));
    // Shared CI runners (esp. macOS) have noisy wall-clock; keep a loose
    // bound that still fails if the cache is completely ineffective.
    REQUIRE(dur < base_duration * 50);
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
      auto [e2, f2, v2] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      REQUIRE_THAT(e2, WithinAbs(e_base, 1e-12));
      REQUIRE(dur < base_duration * 50);
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
      auto [e_read, f_read, v_read] = (*pot)(positions, types, box);
      auto end = high_resolution_clock::now();
      auto dur = duration_cast<nanoseconds>(end - start).count();

      // Should be a Hit (fast) despite being a new object
      REQUIRE_THAT(e_read, WithinAbs(e_base, 1e-12));
      // Timing is noisy on CI (esp. macOS/shared runners); keep a loose bound
      // so we still catch pathological misses (orders of magnitude slower).
      REQUIRE(dur < base_duration * 50);
    }
  }

  SECTION("Uninitialized Cache (Graceful Degradation)") {
    // Cache object created but no DB set
    auto pcache_empty = rgpot::cache::PotentialCache();
    pot->set_cache(&pcache_empty);

    // Should call forceImpl directly without crashing
    auto start = high_resolution_clock::now();
    auto [e, f, v] = (*pot)(positions, types, box);
    auto end = high_resolution_clock::now();

    REQUIRE_THAT(e, WithinAbs(e_base, 1e-12));
    // Should NOT be faster than base (it effectively IS base overhead)
    // We just check it didn't throw exceptions
  }
}

TEST_CASE("Cache keys separate parameter sets", "[Potential][cache]") {
  // Two atoms 5 A apart: inside a 15 A cutoff, outside a 3 A cutoff, so
  // the energies must differ and a shared cache entry would be visible.
  rgpot::types::AtomMatrix positions(2, 3);
  positions(0, 0) = 0.0;
  positions(0, 1) = 0.0;
  positions(0, 2) = 0.0;
  positions(1, 0) = 5.0;
  positions(1, 1) = 0.0;
  positions(1, 2) = 0.0;
  std::vector<int> types(2, 1);
  std::array<std::array<double, 3>, 3> box = {
      {{50, 0, 0}, {0, 50, 0}, {0, 0, 50}}};

  auto potWide = std::make_shared<rgpot::LJPot>();
  auto potNarrow =
      std::make_shared<rgpot::LJPot>(rgpot::LJConfig{.cutoff = 3.0});

  REQUIRE(potWide->paramsKey() != potNarrow->paramsKey());

  std::string db_path = "/tmp/rgpot_test_rocksdb_params";
  rocksdb::Options opts;
  rocksdb::DestroyDB(db_path, opts);
  auto pcache = rgpot::cache::PotentialCache(db_path);
  potWide->set_cache(&pcache);
  potNarrow->set_cache(&pcache);

  auto [eWide, fWide, vWide] = (*potWide)(positions, types, box);
  // Same positions, types, box, and PotType: only paramsKey separates the
  // two entries. A shared entry would return eWide here.
  auto [eNarrow, fNarrow, vNarrow] = (*potNarrow)(positions, types, box);

  REQUIRE(eWide != eNarrow);
  REQUIRE(eNarrow == 0.0); // pair beyond the 3 A cutoff contributes nothing

  // Both entries persist independently.
  auto [eWide2, fWide2, vWide2] = (*potWide)(positions, types, box);
  auto [eNarrow2, fNarrow2, vNarrow2] = (*potNarrow)(positions, types, box);
  REQUIRE(eWide2 == eWide);
  REQUIRE(eNarrow2 == eNarrow);
}
