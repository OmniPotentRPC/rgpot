#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>

namespace rgpot {

/// Open-boundary Sutton-Chen over every ``i < j`` pair.
///
/// Positions are transposed into a reused SoA buffer. The first pass
/// accumulates the densities and the repulsive forces from one integer
/// power of ``(a/r)^2``. The second pass applies the embedding weight.
/// A later call of the same size does not allocate. ``strain`` is the
/// Voigt ``dE/dstrain`` (xx yy zz yz xz xy), or null. Forces are added
/// to the caller buffer. ``energy`` is assigned.
void sutton_chen_direct(const double *pos, std::size_t n, double epsilon,
                        double c, double a, int n_exp, int m_exp, double cutoff,
                        double *forces, double *energy, double *strain);

} // namespace rgpot
