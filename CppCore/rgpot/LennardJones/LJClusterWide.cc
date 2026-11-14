// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exact_fma.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace {

int chunkCount(std::int64_t n) { return nlist::fixedChunks(n); }

struct ChunkScratch {
  std::vector<int> begin;
  std::vector<int> end;
  std::vector<int> origin;
  std::vector<double> slab;
  std::vector<double> partial;
  std::vector<std::size_t> cursor;
  std::uint64_t built = 0;
  std::int64_t atoms = -1;
  bool listed = false;
  bool ready = false;
};

// Equal atom slices leave the low indices with almost every partner when
// the cutoff spans the box. Contiguous slices with the same number of
// higher partners keep the chunk order, and one thread still walks them
// in that order.
void balanceChunks(const std::int32_t *off, const std::int32_t *nbr, std::int64_t n, int k,
                   int *begin, int *end) {
  std::vector<int> run(static_cast<std::size_t>(n) + 1, 0);
  for (std::int64_t i = 0; i < n; ++i) {
    int higher = 0;
    const auto row = static_cast<std::size_t>(i);
    for (std::int32_t p = off[row]; p < off[row + 1]; ++p) {
      if (nbr[static_cast<std::size_t>(p)] > static_cast<std::int32_t>(i)) {
        ++higher;
      }
    }
    run[static_cast<std::size_t>(i) + 1] = run[static_cast<std::size_t>(i)] + higher;
  }
  const int total = run[static_cast<std::size_t>(n)];
  if (total == 0) {
    for (int c = 0; c < k; ++c) {
      const auto range = nlist::partition(n, c, k);
      begin[c] = static_cast<int>(range.begin);
      end[c] = static_cast<int>(range.end);
    }
    return;
  }
  int prev = 0;
  for (int c = 0; c < k; ++c) {
    int edge = static_cast<int>(n);
    if (c + 1 < k) {
      const int goal =
          static_cast<int>((static_cast<long long>(c + 1) * static_cast<long long>(total)) / k);
      int i = prev;
      while (i < static_cast<int>(n) && run[static_cast<std::size_t>(i)] < goal) {
        ++i;
      }
      const int remain = k - c - 1;
      if (static_cast<int>(n) - i < remain) {
        i = static_cast<int>(n) - remain;
      }
      if (i <= prev) {
        i = std::min(static_cast<int>(n), prev + 1);
      }
      edge = i;
    }
    begin[c] = prev;
    end[c] = edge;
    prev = edge;
  }
}

[[gnu::noinline]] void pairTerm(const simd::LjParams &params, double r2, double &pair_e,
                                double &fscale) {
  const double inv_r2 = 1.0 / r2;
  const double sr2 = params.psi2 * inv_r2;
  const double a = (sr2 * sr2) * sr2;
  const double b = params.four_u0 * a;
  if (params.use_switch == 0) {
    pair_e = b * (a - 1.0) - params.shift;
    fscale = ((6.0 * b) * inv_r2) * ((2.0 * a) - 1.0);
    return;
  }
  const double r = std::sqrt(r2);
  const double t_raw = (r - params.r_on) * params.inv_width;
  const double t = t_raw < 0.0 ? 0.0 : (t_raw > 1.0 ? 1.0 : t_raw);
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double poly = simd::fma_quick(t, 15.0 - 6.0 * t, -10.0);
  const double s = simd::fma_quick(t3, poly, 1.0);
  const double dsdr = ((-30.0 * t2) * (1.0 - t) * (1.0 - t)) * params.inv_width;
  const double v = b * (a - 1.0);
  const double f_un = ((6.0 * b) * inv_r2) * ((2.0 * a) - 1.0);
  pair_e = v * s;
  fscale = f_un * s - (v * dsdr) / r;
}

// The image walk still calls this. Inlining it there rebuilds that walk
// and moves every later kernel, including the stored EAM fold.
[[gnu::noinline]] void applyPair(double *local, int origin, std::int64_t i, std::int64_t j,
                                 double dx, double dy, double dz, double r2, double cutoff2,
                                 const simd::LjParams &params, double &energy) {
  if (!(r2 > 0.0) || r2 > cutoff2) {
    return;
  }
  double pair_e = 0.0;
  double fscale = 0.0;
  pairTerm(params, r2, pair_e, fscale);
  energy += pair_e;
  const double fx = fscale * dx;
  const double fy = fscale * dy;
  const double fz = fscale * dz;
  double *fi = local + (static_cast<std::size_t>(i - origin) * 3);
  double *fj = local + (static_cast<std::size_t>(j - origin) * 3);
  fi[0] += fx;
  fi[1] += fy;
  fi[2] += fz;
  fj[0] -= fx;
  fj[1] -= fy;
  fj[2] -= fz;
}

} // namespace

void ljClusterChunks(nlist::Neighbours &cells, const simd::LjParams &params, double cutoff2,
                     double *force, double *energy, std::int64_t n, int nt) {
  if (nt < 1) {
    nt = 1;
  }
  const int k = chunkCount(n);
  const bool plain = cells.rawSeparation();
  const double *pos = cells.coordinates();
  const std::int32_t *off = cells.offsets().data();
  const std::int32_t *nbr = cells.neighbours().data();
  thread_local ChunkScratch scratch;
  scratch.begin.resize(static_cast<std::size_t>(k));
  scratch.end.resize(static_cast<std::size_t>(k));
  scratch.origin.resize(static_cast<std::size_t>(k));
  scratch.partial.assign(static_cast<std::size_t>(k), 0.0);
  const bool listed =
      !cells.streaming() && cells.offsets().size() == static_cast<std::size_t>(n) + 1;
  if (!scratch.ready || scratch.built != cells.builds() || scratch.atoms != n ||
      scratch.listed != listed) {
    if (listed) {
      balanceChunks(off, nbr, n, k, scratch.begin.data(), scratch.end.data());
    } else {
      for (int c = 0; c < k; ++c) {
        const auto range = nlist::partition(n, c, k);
        scratch.begin[static_cast<std::size_t>(c)] = static_cast<int>(range.begin);
        scratch.end[static_cast<std::size_t>(c)] = static_cast<int>(range.end);
      }
    }
    scratch.built = cells.builds();
    scratch.atoms = n;
    scratch.listed = listed;
    scratch.ready = true;
  }
  std::size_t words = 0;
  for (int c = 0; c < k; ++c) {
    const int edge = scratch.begin[static_cast<std::size_t>(c)];
    // A half pair i < j writes +f at i and -f at j, so the buffer starts
    // at the first i of the chunk and runs to the last atom.
    scratch.origin[static_cast<std::size_t>(c)] = edge;
    const std::size_t span = scratch.begin[static_cast<std::size_t>(c)] ==
                                     scratch.end[static_cast<std::size_t>(c)]
                                 ? 0
                                 : static_cast<std::size_t>(n - edge);
    words += span * 3;
  }
  scratch.slab.assign(words, 0.0);
  int *begin = scratch.begin.data();
  int *end = scratch.end.data();
  int *origin = scratch.origin.data();
  double *slab = scratch.slab.data();
  double *partial = scratch.partial.data();
  scratch.cursor.assign(static_cast<std::size_t>(k) + 1, 0);
  std::vector<std::size_t> &cursor = scratch.cursor;
  for (int c = 0; c < k; ++c) {
    const std::size_t span =
        begin[c] == end[c] ? 0 : static_cast<std::size_t>(n - origin[c]);
    cursor[static_cast<std::size_t>(c) + 1] =
        cursor[static_cast<std::size_t>(c)] + span * 3;
  }
  std::size_t *base = cursor.data();
  auto run_chunk = [&](int c) {
      const std::int64_t i0 = begin[c];
      const std::int64_t i1 = end[c];
      if (i0 == i1) {
        return;
      }
      double *local = slab + base[c];
      const int org = origin[c];
      double local_e = 0.0;
      if (plain) {
        for (std::int64_t i = i0; i < i1; ++i) {
          const auto ui = static_cast<std::size_t>(i) * 3;
          const double xi = pos[ui];
          const double yi = pos[ui + 1];
          const double zi = pos[ui + 2];
          const auto row = static_cast<std::size_t>(i);
          const std::int32_t *const row_nbr = nbr + off[row];
          const std::int32_t *const row_end = nbr + off[row + 1];
          // Partners are sorted, so the half list is the suffix j > i.
          for (const std::int32_t *it =
                   std::upper_bound(row_nbr, row_end, static_cast<std::int32_t>(i));
               it != row_end; ++it) {
            const auto j = static_cast<std::int64_t>(*it);
            const auto uj = static_cast<std::size_t>(j) * 3;
            const double dx = xi - pos[uj];
            const double dy = yi - pos[uj + 1];
            const double dz = zi - pos[uj + 2];
            const double r2 = (dx * dx + dy * dy) + dz * dz;
            // The vacuum walk is the hot one. The arithmetic sits here so
            // it is not a separate call; the image walk keeps that call.
            if (!(r2 > 0.0) || r2 > cutoff2) {
              continue;
            }
            const double inv_r2 = 1.0 / r2;
            const double sr2 = params.psi2 * inv_r2;
            const double a = (sr2 * sr2) * sr2;
            const double b = params.four_u0 * a;
            double pair_e = 0.0;
            double fscale = 0.0;
            if (params.use_switch == 0) {
              pair_e = b * (a - 1.0) - params.shift;
              fscale = ((6.0 * b) * inv_r2) * ((2.0 * a) - 1.0);
            } else {
              pairTerm(params, r2, pair_e, fscale);
            }
            local_e += pair_e;
            const double fx = fscale * dx;
            const double fy = fscale * dy;
            const double fz = fscale * dz;
            double *fi = local + (static_cast<std::size_t>(i - org) * 3);
            double *fj = local + (static_cast<std::size_t>(j - org) * 3);
            fi[0] += fx;
            fi[1] += fy;
            fi[2] += fz;
            fj[0] -= fx;
            fj[1] -= fy;
            fj[2] -= fz;
          }
        }
      } else {
        // One walk for the chunk. The image tables are built once here;
        // a call per atom rebuilt them. j > i is the same suffix as before.
        cells.forDirected(i0, i1, [&](std::int32_t i, std::int32_t j, double dx, double dy,
                                      double dz, double r2) {
          if (static_cast<std::int64_t>(j) <= static_cast<std::int64_t>(i)) {
            return;
          }
          applyPair(local, org, static_cast<std::int64_t>(i), static_cast<std::int64_t>(j), dx,
                    dy, dz, r2, cutoff2, params, local_e);
        });
      }
      partial[c] = local_e;
  };
  if (nt == 1) {
    for (int c = 0; c < k; ++c) {
      run_chunk(c);
    }
  } else {
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
#endif
    {
#if defined(_OPENMP)
      const int thread = omp_get_thread_num();
#else
      const int thread = 0;
#endif
      for (int c = thread; c < k; c += nt) {
        run_chunk(c);
      }
    }
  }
  double total = 0.0;
  for (int c = 0; c < k; ++c) {
    if (begin[c] == end[c]) {
      continue;
    }
    total += partial[c];
    const double *local = slab + base[c];
    const int org = origin[c];
    for (std::int64_t a = org; a < n; ++a) {
      const double *src = local + static_cast<std::size_t>(a - org) * 3;
      double *dst = force + static_cast<std::size_t>(a) * 3;
      dst[0] += src[0];
      dst[1] += src[1];
      dst[2] += src[2];
    }
  }
  *energy += total;
}

} // namespace rgpot
