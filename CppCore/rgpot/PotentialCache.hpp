#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Header file for the PotentialCache class.
 *
 * This file defines the caching mechanism for the rgpot library, utilizing
 * RocksDB to store and retrieve potential energy and force calculations.
 */

#include "rgpot/types/AtomMatrix.hpp"
#include <optional>
#include <rocksdb/db.h>
#include <string>

namespace rgpot::cache {

/**
 * @class KeyHash
 * @brief Struct to hold the hash and string key for caching.
 * @ingroup rgpot_cache
 */
struct KeyHash {
  size_t hash;     //!< The numeric hash value.
  std::string key; //!< The string representation of the hash.

  /**
   * @brief Constructor for KeyHash.
   * @param _hash The numeric hash to wrap.
   */
  KeyHash(size_t _hash) : hash{_hash}, key(std::to_string(_hash)) {}
};

/**
 * @class PotentialCache
 * @brief Caches potential energy and force calculations using RocksDB.
 * @ingroup rgpot_cache
 */
class PotentialCache {
private:
  rocksdb::DB *db_ = nullptr; //!< Pointer to the RocksDB instance.
  bool own_db_ = false;       //!< Ownership flag for the DB pointer.

public:
  /**
   * @brief Constructor opens the DB at the given path.
   * @param db_path Path to the RocksDB database.
   * @param create_if_missing Toggle creation of DB if absent.
   */
  explicit PotentialCache(const std::string &db_path,
                          bool create_if_missing = true);

  /**
   * @brief Default constructor.
   */
  PotentialCache() = default;

  /**
   * @brief Destructor.
   */
  ~PotentialCache();

  /**
   * @brief Helper for manual pointer setting.
   * @param db Pointer to an existing RocksDB instance.
   * @return Void.
   */
  void set_db(rocksdb::DB *db);

  /**
   * @brief Deserializes a cache hit into output containers.
   * @param value Serialized string from the cache.
   * @param energy Reference to store the energy.
   * @param forces Reference to store the forces.
   * @return Void.
   */
  void deserialize_hit(const std::string &value, double &energy,
                       rgpot::types::AtomMatrix &forces);

  /**
   * @brief Adds a serialized calculation to the cache.
   * @param key Unique hash key for the configuration.
   * @param energy Calculated energy.
   * @param forces Calculated forces.
   * @return Void.
   */
  void add_serialized(const KeyHash &key, double energy,
                      const rgpot::types::AtomMatrix &forces);

  /**
   * @brief Searches the cache for a specific key.
   * @param key Unique hash key.
   * @return Optional string containing the serialized data.
   */
  std::optional<std::string> find(const KeyHash &key);
};

} // namespace rgpot::cache
