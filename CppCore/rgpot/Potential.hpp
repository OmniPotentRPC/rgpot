#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

/**
 * @brief Base classes and templates for chemical potentials.
 *
 * Provides the abstract interface and CRTP template for all potential energy
 * surfaces. Handles the high-level logic for caching, hashing, and force call
 * registration.
 */

// clang-format off
#include <utility>
#include <vector>
#include <stdexcept>
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

/**
 * @class PotentialBase
 * @brief Abstract base class for all potential energy surfaces.
 */
class PotentialBase {
public:
  /**
   * @brief Constructor for PotentialBase.
   * @param inp_type The type of the potential.
   */
  explicit PotentialBase(PotType inp_type) : m_type(inp_type) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~PotentialBase() = default;

  /**
   * @brief Main interface for potential and force calculation.
   * @param positions The atomic coordinates.
   * @param atmtypes The atomic numbers.
   * @param box The simulation cell vectors.
   * @return A pair containing the energy and the force matrix.
   */
  virtual std::pair<double, AtomMatrix>
  operator()(const AtomMatrix &positions, const std::vector<int> &atmtypes,
             const std::array<std::array<double, 3>, 3> &box) = 0;

#ifdef POT_HAS_CACHE
  /**
   * @brief Sets the computation cache.
   * @param c Pointer to a PotentialCache instance.
   * @return Void.
   */
  virtual void set_cache(rgpot::cache::PotentialCache * /*c*/) {
    throw std::runtime_error("PotentialBase::set_cache called directly");
  }
#endif

  /**
   * @brief Fetches the potential type.
   * @return The potential type.
   */
  [[nodiscard]] PotType get_type() const { return m_type; }

protected:
  PotType m_type; //!< The type of the potential energy surface.
};

/**
 * @class Potential
 * @brief Template class for specific potential implementations.
 *
 * Uses the Curiously Recurring Template Pattern to provide static
 * polymorphism for the internal @c forceImpl call.
 */
template <typename Derived>
class Potential : public PotentialBase, public registry<Derived> {
public:
  using PotentialBase::PotentialBase;

#ifdef POT_HAS_CACHE
  /**
   * @brief Sets the computation cache for the specific implementation.
   * @param c Pointer to a PotentialCache instance.
   * @return Void.
   */
  void set_cache(rgpot::cache::PotentialCache *c) override { _cache = c; }
#endif

  /**
   * @brief Implements the potential and force calculation logic.
   *
   * This method manages the transformation of @c Eigen matrices into
   * flat @c double arrays.
   *
   * # Caching Logic
   * If @c POT_HAS_CACHE is defined, the method:
   * 1. Generates a @c XXH3_64bits hash of positions, types, and box.
   * 2. Checks the @c rocksdb backend for a hit.
   * 3. Returns cached values if present, otherwise computes and stores results.
   *
   * @param positions The atomic coordinates.
   * @param atmtypes The atomic numbers.
   * @param box The simulation cell vectors.
   * @return A pair containing the energy and the force matrix.
   */
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

  /**
   * @brief Abstract hook for the actual implementation.
   * @param in Structure containing coordinates and cell info.
   * @param out Pointer to the results structure.
   * @return Void.
   */
  virtual void forceImpl(const ForceInput &in, ForceOut *out) const = 0;

private:
#ifdef POT_HAS_CACHE
  rgpot::cache::PotentialCache *_cache =
      nullptr; //!< Pointer to the optional calculation cache.
#endif
};

} // namespace rgpot
