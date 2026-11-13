// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the RocksDB-based potential cache.
 *
 * This file implements the methods for the PotentialCache class, handling the
 * low-level interactions with the RocksDB library for persistent storage of
 * potential evaluations.
 */

#include "rgpot/PotentialCache.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <rocksdb/options.h>
#include <stdexcept>
#include <vector>

namespace rgpot::cache {

namespace {
constexpr char result_marker[] = "RGPOTR1";
constexpr char virial_marker[] = "RGPOTR2";
constexpr size_t result_metadata_bytes =
    sizeof(result_marker) + 10 * sizeof(double) + 1;
// marker, variance, stress[9], virial[9], has_stress, has_virial.
constexpr size_t virial_metadata_bytes =
    sizeof(virial_marker) + 19 * sizeof(double) + 2;

size_t energy_force_bytes(size_t n_atoms) {
  return (1 + 3 * n_atoms) * sizeof(double);
}

bool marker_at(const std::string &value, size_t prefix, const char *marker,
               size_t marker_bytes) {
  return value.size() >= prefix + marker_bytes &&
         std::memcmp(value.data() + prefix, marker, marker_bytes) == 0;
}
} // namespace

bool PotentialCache::has_result_metadata(const std::string &value,
                                         size_t n_atoms) {
  const size_t prefix = energy_force_bytes(n_atoms);
  if (value.size() == prefix + result_metadata_bytes &&
      marker_at(value, prefix, result_marker, sizeof(result_marker)) &&
      static_cast<unsigned char>(value.back()) <= 1) {
    return true;
  }
  return has_virial_metadata(value, n_atoms);
}

bool PotentialCache::has_virial_metadata(const std::string &value,
                                         size_t n_atoms) {
  const size_t prefix = energy_force_bytes(n_atoms);
  if (value.size() != prefix + virial_metadata_bytes ||
      !marker_at(value, prefix, virial_marker, sizeof(virial_marker))) {
    return false;
  }
  const auto stress_flag = static_cast<unsigned char>(
      value[value.size() - 2]);
  const auto virial_flag = static_cast<unsigned char>(value.back());
  return stress_flag <= 1 && virial_flag <= 1;
}

void PotentialCache::deserialize_hit(const std::string &value, ForceOut &out,
                                     size_t n_atoms) {
  const size_t prefix = energy_force_bytes(n_atoms);
  if (value.size() < prefix) {
    throw std::runtime_error("PotentialCache: incomplete energy/force record");
  }
  std::memcpy(&out.energy, value.data(), sizeof(double));
  std::memcpy(out.F, value.data() + sizeof(double),
              3 * n_atoms * sizeof(double));
  out.variance = 0.0;
  out.has_stress = 0;
  out.has_virial = 0;
  std::fill(out.stress, out.stress + 9, 0.0);
  std::fill(out.virial, out.virial + 9, 0.0);
  if (has_virial_metadata(value, n_atoms)) {
    const char *metadata = value.data() + prefix + sizeof(virial_marker);
    std::memcpy(&out.variance, metadata, sizeof(double));
    std::memcpy(out.stress, metadata + sizeof(double), sizeof(out.stress));
    std::memcpy(out.virial, metadata + 10 * sizeof(double), sizeof(out.virial));
    out.has_stress = static_cast<unsigned char>(value[value.size() - 2]);
    out.has_virial = static_cast<unsigned char>(value.back());
  } else if (has_result_metadata(value, n_atoms)) {
    const char *metadata = value.data() + prefix + sizeof(result_marker);
    std::memcpy(&out.variance, metadata, sizeof(double));
    std::memcpy(out.stress, metadata + sizeof(double), sizeof(out.stress));
    out.has_stress = static_cast<unsigned char>(value.back());
  }
}

void PotentialCache::add_serialized(const KeyHash &key, const ForceOut &out,
                                    size_t n_atoms) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (!db_) {
    return;
  }
  const size_t prefix = energy_force_bytes(n_atoms);
  std::vector<char> buffer(prefix + virial_metadata_bytes, 0);
  std::memcpy(buffer.data(), &out.energy, sizeof(double));
  std::memcpy(buffer.data() + sizeof(double), out.F,
              3 * n_atoms * sizeof(double));
  std::memcpy(buffer.data() + prefix, virial_marker, sizeof(virial_marker));
  char *metadata = buffer.data() + prefix + sizeof(virial_marker);
  std::memcpy(metadata, &out.variance, sizeof(double));
  if (out.has_stress) {
    std::memcpy(metadata + sizeof(double), out.stress, sizeof(out.stress));
    buffer[buffer.size() - 2] = 1;
  }
  if (out.has_virial) {
    std::memcpy(metadata + 10 * sizeof(double), out.virial, sizeof(out.virial));
    buffer.back() = 1;
  }
  db_->Put(rocksdb::WriteOptions(), key.slice(),
           rocksdb::Slice(buffer.data(), buffer.size()));
}

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
#if RGPOT_ROCKSDB_UNIQUE_PTR_OPEN
  std::unique_ptr<rocksdb::DB> db;
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
  db_ = db.release();
#else
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_);
#endif
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
  std::lock_guard<std::mutex> lock(db_mu_);
  if (own_db_ && db_) {
    delete db_;
    db_ = nullptr;
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
  std::lock_guard<std::mutex> lock(db_mu_);
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
  ForceOut out{.F = forces.data()};
  deserialize_hit(hit, out, static_cast<size_t>(forces.rows()));
  energy = out.energy;
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
  std::lock_guard<std::mutex> lock(db_mu_);
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
  db_->Put(rocksdb::WriteOptions(), kv.slice(), value);
}

/**
 * @details
 * Queries the RocksDB instance for the given key.
 *
 * @return std::optional containing the serialized string if found,
 * or std::nullopt if the key does not exist or the DB is closed.
 */
std::optional<std::string> PotentialCache::find(const KeyHash &kv) {
  std::lock_guard<std::mutex> lock(db_mu_);
  if (!db_)
    return std::nullopt;
  std::string value;
  rocksdb::Status s = db_->Get(rocksdb::ReadOptions(), kv.slice(), &value);
  if (s.ok()) {
    return value;
  }
  return std::nullopt;
}
} // namespace rgpot::cache
