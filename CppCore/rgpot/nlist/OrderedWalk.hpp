#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "rgpot/nlist/Neighbours.hpp"

namespace rgpot::nlist {

/// One half-list pair, in the order `forHalf(0, n)` visits it.
/// Replaying the thread slices in thread order is that same order: each
/// slice is a contiguous run of atoms.
struct HalfTerm {
  std::int32_t i = 0;
  std::int32_t j = 0;
  double e = 0;
  double fscale = 0;
  double dx = 0;
  double dy = 0;
  double dz = 0;
};

struct alignas(64) HalfPart {
  std::vector<HalfTerm> terms;
};

/// `fill` returns false when the pair is outside the kernel. The kept
/// pairs of thread `t` are the half list of atoms `[begin, end)`.
template <typename Fill>
void gather_half(const Neighbours &cells, int nt, std::int64_t n,
                 std::vector<HalfPart> &parts, Fill fill) {
  if (nt < 1) {
    nt = 1;
  }
  parts.assign(static_cast<std::size_t>(nt), {});
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
  {
#if defined(_OPENMP)
    const int thread = omp_get_thread_num();
#else
    const int thread = 0;
#endif
    const Range range = partition(n, thread, nt);
    auto &mine = parts[static_cast<std::size_t>(thread)].terms;
    const auto span = range.end - range.begin;
    if (span > 0) {
      mine.reserve(static_cast<std::size_t>(span) * 16);
    }
    cells.forHalf(range.begin, range.end,
                  [&](std::int32_t i, std::int32_t j, double dx, double dy, double dz,
                      double r2) {
                    HalfTerm term;
                    term.i = i;
                    term.j = j;
                    term.dx = dx;
                    term.dy = dy;
                    term.dz = dz;
                    if (fill(i, j, dx, dy, dz, r2, term)) {
                      mine.push_back(term);
                    }
                  });
  }
}

/// Add pair energy and forces in the gathered order. `strain(term)` runs
/// after the force update, in that same order.
template <typename Strain>
void replay_half(const std::vector<HalfPart> &parts, double *force, double &energy,
                 Strain strain) {
  for (const HalfPart &part : parts) {
    for (const HalfTerm &term : part.terms) {
      energy += term.e;
      const double fx = term.fscale * term.dx;
      const double fy = term.fscale * term.dy;
      const double fz = term.fscale * term.dz;
      force[3 * term.i] += fx;
      force[3 * term.i + 1] += fy;
      force[3 * term.i + 2] += fz;
      force[3 * term.j] -= fx;
      force[3 * term.j + 1] -= fy;
      force[3 * term.j + 2] -= fz;
      strain(term);
    }
  }
}

} // namespace rgpot::nlist
