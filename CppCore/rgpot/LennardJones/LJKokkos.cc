// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/LennardJones/LJKokkos.hpp"

#include "kokkos_rs.h"

#include <stdexcept>

namespace rgpot {

LJKokkos::LJKokkos(double u0, double psi, double cutoff, bool periodic,
                   std::size_t threads, std::size_t min_atoms)
    : k_(nullptr) {
  krs_lj_config cfg{};
  cfg.u0 = u0;
  cfg.psi = psi;
  cfg.cutoff = cutoff;
  cfg.skin = 1.0;
  cfg.threads = threads;
  cfg.min_atoms = min_atoms;
  cfg.periodic = periodic ? 1 : 0;
  k_ = krs_lj_new(&cfg);
  if (k_ == nullptr) {
    throw std::runtime_error("krs_lj_new failed");
  }
}

LJKokkos::~LJKokkos() { krs_lj_free(k_); }

void LJKokkos::forceImpl(const ForceInput &in, ForceOut *out) {
  const int rc = krs_lj_force(k_, in.nAtoms, in.pos, in.box, &out->energy, out->F);
  if (rc != 0) {
    throw std::runtime_error("krs_lj_force failed");
  }
  out->variance = 0.0;
  out->has_stress = 0;
}

std::uint64_t LJKokkos::forceCalls() const { return krs_lj_force_calls(k_); }

std::uint64_t LJKokkos::pairCount() const {
  krs_traffic traffic{};
  krs_lj_last_traffic(k_, &traffic);
  return traffic.pairs;
}

std::uint64_t LJKokkos::tileBytes() const {
  krs_traffic traffic{};
  krs_lj_last_traffic(k_, &traffic);
  return traffic.tile_bytes;
}

std::uint32_t LJKokkos::threadsUsed() const {
  krs_traffic traffic{};
  krs_lj_last_traffic(k_, &traffic);
  return traffic.threads_used;
}

} // namespace rgpot
