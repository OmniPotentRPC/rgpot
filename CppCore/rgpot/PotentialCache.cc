// MIT License
// Copyright 2026--present Rohit Goswami <HaoZeke>
#include "rgpot/PotentialCache.hpp"
#include <cstring>
#include <vector>

namespace rgpot::cache {
void PotentialCache::set_cache(rocksdb::DB *cc) { potCache = cc; }

void PotentialCache::deserialize_hit(const std::string &hit, double &energy,
                                     rgpot::types::AtomMatrix &forces) {
  // hit contains the raw binary buffer
  std::memcpy(&energy, hit.data(), sizeof(double));
  std::memcpy(forces.data(), hit.data() + sizeof(double),
              forces.size() * sizeof(double));
}

void PotentialCache::add_serialized(const KeyHash &kv, double energy,
                                    const rgpot::types::AtomMatrix &forces) {
  size_t value_size = sizeof(double) + forces.size() * sizeof(double);
  std::vector<char> buffer(value_size);
  std::memcpy(buffer.data(), &energy, sizeof(double));
  std::memcpy(buffer.data() + sizeof(double), forces.data(),
              forces.size() * sizeof(double));

  rocksdb::Slice value(buffer.data(), buffer.size());
  potCache->Put(rocksdb::WriteOptions(), kv.key, value);
}

std::optional<std::string> PotentialCache::find(const KeyHash &kv) {
  std::string value;
  rocksdb::Status s = potCache->Get(rocksdb::ReadOptions(), kv.key, &value);
  if (s.ok()) {
    return value;
  }
  return std::nullopt;
}
} // namespace rgpot::cache
