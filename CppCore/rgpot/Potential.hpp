#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
// clang-format off
#include <utility>
#include <vector>
#include<stdexcept>
// clang-format on

#ifdef POT_HAS_CACHE
#define XXH_INLINE_ALL
#include "rgpot/PotentialCache.hpp"
#include <xxhash.h>
#endif

#include "rgpot/ForceStructs.hpp"
#include "rgpot/PotHelpers.hpp"
#include "rgpot/pot_types.hpp"
#include "rgpot/types/AtomMatrix.hpp"

using rgpot::types::AtomMatrix;

namespace rgpot {

class PotentialBase {
public:
  explicit PotentialBase(PotType inp_type) : m_type(inp_type) {}
  virtual ~PotentialBase() = default;

  virtual std::pair<double, AtomMatrix>
  operator()(const AtomMatrix &positions, const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) = 0;

#ifdef POT_HAS_CACHE
  virtual void set_cache(rgpot::cache::PotentialCache * /*c*/) {
    throw std::runtime_error("PotentialBase::set_cache called directly");
  }
#endif
  [[nodiscard]] PotType get_type() const { return m_type; }

protected:
  PotType m_type;
};

template <typename Derived>
class Potential : public PotentialBase, public registry<Derived> {
public:
  using PotentialBase::PotentialBase;

#ifdef POT_HAS_CACHE
  void set_cache(rgpot::cache::PotentialCache *c) override { _cache = c; }
#endif

  std::pair<double, AtomMatrix>
  operator()(const AtomMatrix &positions, const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) override {
    size_t nAtoms = positions.rows();
    AtomMatrix forces = AtomMatrix::Zero(nAtoms, 3);
    double energy = 0.0;
    double variance = 0.0;

    double flatBox[9];
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        flatBox[i * 3 + j] = box[i][j];
      }
    }

    ForceInput fi{.nAtoms = nAtoms,
                  .pos = positions.data(),
                  .atmnrs = atmtypes.data(),
                  .box = flatBox};
    ForceOut fo{.F = forces.data(), .energy = energy, .variance = variance};

#ifdef POT_HAS_CACHE
    // Hashing
    size_t hash_val = 0;
    hash_val ^= XXH3_64bits(fi.pos, fi.nAtoms * 3 * sizeof(double));
    hash_val ^= XXH3_64bits(fi.atmnrs, fi.nAtoms * sizeof(int));
    hash_val ^= XXH3_64bits(fi.box, 9 * sizeof(double));
    size_t type_val = static_cast<size_t>(m_type);
    hash_val ^= XXH3_64bits(&type_val, sizeof(size_t));

    rgpot::cache::KeyHash key(hash_val);

    // Cache Read
    if (_cache) {
      auto hit = _cache->find(key);
      if (hit) {
        _cache->deserialize_hit(*hit, fo.energy, forces);
        return {fo.energy, forces};
      }
    }

    // Computation
    static_cast<Derived *>(this)->forceImpl(fi, &fo);
    registry<Derived>::incrementForceCalls();

    // Cache Write
    if (_cache) {
      _cache->add_serialized(key, fo.energy, forces);
    }
#else
    // Fallback when caching is disabled
    static_cast<Derived *>(this)->forceImpl(fi, &fo);
    registry<Derived>::incrementForceCalls();
#endif

    return {fo.energy, forces};
  }

  // Virtual to allow Python override if strictly necessary
  virtual void forceImpl(const ForceInput &in, ForceOut *out) const = 0;

private:
#ifdef POT_HAS_CACHE
  rgpot::cache::PotentialCache *_cache = nullptr;
#endif
};

} // namespace rgpot
