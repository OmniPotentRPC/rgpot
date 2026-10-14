#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * Lennard-Jones force entry evaluated by kokkos-rs.
 *
 * The pair term is the shifted 12-6 used by LJPot. `periodic` selects
 * the orthorhombic minimum image; the free cluster passes false, matching
 * LJClusterPot. The host pool width is `threads`. Fewer than `min_atoms`
 * atoms run on the calling thread.
 */

#include "rgpot/ForceStructs.hpp"

#include <cstddef>
#include <cstdint>

struct krs_lj;
struct krs_traffic;

namespace rgpot {

class LJKokkos {
public:
  LJKokkos(double u0, double psi, double cutoff, bool periodic,
           std::size_t threads, std::size_t min_atoms = 0);
  ~LJKokkos();
  LJKokkos(const LJKokkos &) = delete;
  LJKokkos &operator=(const LJKokkos &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out);

  [[nodiscard]] std::uint64_t forceCalls() const;
  [[nodiscard]] std::uint64_t pairCount() const;
  [[nodiscard]] std::uint64_t tileBytes() const;
  [[nodiscard]] std::uint32_t threadsUsed() const;

private:
  krs_lj *k_;
};

} // namespace rgpot
