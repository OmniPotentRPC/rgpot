// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Implementation of the RocksDB-based potential cache.
 *
 * This file implements the methods for the PotentialCache class, handling the
 * low-level interactions with the RocksDB library for persistent storage of
 * potential evaluations.
 */

#include "rgpot/PotentialCache.hpp"
#include <cstring>
#include <iostream>
#include <rocksdb/options.h>
#include <vector>

namespace rgpot::cache {

/**
 * @details
 * Initializes the RocksDB options, specifically setting `create_if_missing`.
 * It attempts to open the database at the specified path. If the open fails,
 * an error is printed to stderr and the internal database pointer remains null.
 * If successful, the `own_db_` flag is set to true.
 */
PotentialCache::PotentialCache(const std::string &db_path,
                               bool create_if_missing) {
  rocksdb::Options options;
  options.create_if_missing = create_if_missing;
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_);
  if (!status.ok()) {
    std::cerr << "Unable to open RocksDB at " << db_path << ": "
              << status.ToString() << std::endl;
    db_ = nullptr;
  } else {
    own_db_ = true;
  }
}

/**
 * @details
 * Checks if the class owns the database pointer. If so, it deletes the
 * `rocksdb::DB` instance to prevent memory leaks.
 */
PotentialCache::~PotentialCache() {
  if (own_db_ && db_) {
    delete db_;
  }
}

/**
 * @details
 * This function allows for manually injecting a RocksDB pointer.
 * If the current instance already owns a database, it is deleted before
 * accepting the new pointer. Ownership is transferred away from this class
 * (own_db_ set to false), implying the caller manages the new pointer's
 * lifetime or it is a shared resource.
 *
 * @warning Use with caution to avoid double-free or memory leaks if the
 * passed pointer is managed elsewhere.
 */
void PotentialCache::set_db(rocksdb::DB *db) {
  if (own_db_ && db_)
    delete db_;
  db_ = db;
  own_db_ = false;
}

/**
 * @details
 * Performs a binary copy (`std::memcpy`) from the serialized string buffer
 * back into the energy variable and force matrix.
 *
 * The layout is assumed to be:
 * `[double energy] [double force_0] ... [double force_N]`
 */
void PotentialCache::deserialize_hit(const std::string &hit, double &energy,
                                     rgpot::types::AtomMatrix &forces) {
  std::memcpy(&energy, hit.data(), sizeof(double));
  std::memcpy(forces.data(), hit.data() + sizeof(double),
              forces.size() * sizeof(double));
}

/**
 * @details
 * Serializes the energy and forces into a contiguous binary buffer.
 * The buffer size is calculated as `sizeof(double) + N * sizeof(double)`.
 * This buffer is then stored in RocksDB using the provided key.
 *
 * @note If the database is not initialized, this function returns immediately.
 */
void PotentialCache::add_serialized(const KeyHash &kv, double energy,
                                    const rgpot::types::AtomMatrix &forces) {
  if (!db_)
    return;
  // Calculate total size needed
  size_t value_size = sizeof(double) + forces.size() * sizeof(double);
  std::vector<char> buffer(value_size);

  // Copy energy to buffer
  std::memcpy(buffer.data(), &energy, sizeof(double));
  // Copy forces to buffer offset by sizeof(double)
  std::memcpy(buffer.data() + sizeof(double), forces.data(),
              forces.size() * sizeof(double));

  rocksdb::Slice value(buffer.data(), buffer.size());
  db_->Put(rocksdb::WriteOptions(), kv.key, value);
}

/**
 * @details
 * Queries the RocksDB instance for the given key.
 *
 * @return std::optional containing the serialized string if found,
 * or std::nullopt if the key does not exist or the DB is closed.
 */
std::optional<std::string> PotentialCache::find(const KeyHash &kv) {
  if (!db_)
    return std::nullopt;
  std::string value;
  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), kv.key, &value);
  if (s.ok()) {
    return value;
  }
  return std::nullopt;
}
} // namespace rgpot::cache
