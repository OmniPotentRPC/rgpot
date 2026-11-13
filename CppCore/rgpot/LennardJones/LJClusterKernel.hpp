#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>

namespace rgpot {

/// Open-boundary shifted Lennard-Jones over every ``i < j`` pair.
///
/// Positions are transposed into a reused SoA buffer (no allocation on a
/// later call of the same size). The inner loop is AVX2 on x86_64 when
/// the CPU has it, and NEON on AArch64. Both issue separate multiplies
/// and adds, and the pair contributions are added back in ``i``, then
/// ``j``, order. The translation unit is built with contraction off so
/// those adds stay in that order.
void lj_cluster_direct(const double *pos, std::size_t n, double cutoff, double u0,
                       double psi, double cutoff_u, double *forces, double *energy);

} // namespace rgpot
