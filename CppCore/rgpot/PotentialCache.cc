// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "rgpot/PotentialCache.hpp"
#include <cstring>
#include <iostream>
#include <rocksdb/options.h>
#include <vector>

namespace rgpot::cache {

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

PotentialCache::~PotentialCache() {
  if (own_db_ && db_) {
    delete db_;
  }
}

void PotentialCache::set_db(rocksdb::DB *db) {
  if (own_db_ && db_)
    delete db_;
  db_ = db;
  own_db_ = false;
}

void PotentialCache::deserialize_hit(const std::string &hit, double &energy,
                                     rgpot::types::AtomMatrix &forces) {
  std::memcpy(&energy, hit.data(), sizeof(double));
  std::memcpy(forces.data(), hit.data() + sizeof(double),
              forces.size() * sizeof(double));
}

void PotentialCache::add_serialized(const KeyHash &kv, double energy,
                                    const rgpot::types::AtomMatrix &forces) {
  if (!db_)
    return;
  size_t value_size = sizeof(double) + forces.size() * sizeof(double);
  std::vector<char> buffer(value_size);
  std::memcpy(buffer.data(), &energy, sizeof(double));
  std::memcpy(buffer.data() + sizeof(double), forces.data(),
              forces.size() * sizeof(double));

  rocksdb::Slice value(buffer.data(), buffer.size());
  db_->Put(rocksdb::WriteOptions(), kv.key, value);
}

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
