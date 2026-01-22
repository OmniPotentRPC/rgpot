#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include "rgpot/types/AtomMatrix.hpp"
#include <optional>
#include <rocksdb/db.h>
#include <string>

namespace rgpot::cache {

struct KeyHash {
  size_t hash;
  std::string key;
  KeyHash(size_t _hash) : hash{_hash}, key(std::to_string(_hash)) {}
};

class PotentialCache {
private:
  rocksdb::DB *db_ = nullptr;
  bool own_db_ = false;

public:
  // Constructor opens the DB at the given path
  explicit PotentialCache(const std::string &db_path,
                          bool create_if_missing = true);
  // Default constructor (no cache)
  PotentialCache() = default;
  ~PotentialCache();

  // Helper for manual pointer setting (legacy/testing)
  void set_db(rocksdb::DB *db);

  void deserialize_hit(const std::string &, double &,
                       rgpot::types::AtomMatrix &);
  void add_serialized(const KeyHash &, double,
                      const rgpot::types::AtomMatrix &);
  std::optional<std::string> find(const KeyHash &);
};

} // namespace rgpot::cache
