#pragma once
// MIT License
// Copyright 2026--present Rohit Goswami <HaoZeke>

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
  rocksdb::DB *potCache = nullptr;

public:
  void set_cache(rocksdb::DB *);
  void deserialize_hit(const std::string &, double &,
                       rgpot::types::AtomMatrix &);
  void add_serialized(const KeyHash &, double,
                      const rgpot::types::AtomMatrix &);
  std::optional<std::string> find(const KeyHash &);
};

} // namespace rgpot::cache
