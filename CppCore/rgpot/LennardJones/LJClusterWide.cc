// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/nlist/NeighbourList.hpp"
#include "rgpot/simd/api.hpp"
#include "rgpot/simd/exact_fma.hpp"

#include <cstdint>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rgpot {
namespace {

// Chunk count depends only on the atom count. Threads take whole chunks
// and the output adds chunk 0, then 1, and so on, so the bits do not
// depend on how many threads ran.
int chunkCount(std::int64_t n) {
#if defined(LJC_CHUNK_CAP)
  constexpr int cap = LJC_CHUNK_CAP;
#else
  constexpr int cap = 16;
#endif
  if (n < 2) {
    return 1;
  }
  if (n < cap) {
    return static_cast<int>(n);
  }
  return cap;
}

struct ChunkScratch {
  std::vector<int> begin;
  std::vector<int> end;
  std::vector<int> origin;
  std::vector<double> slab;
  std::vector<double> partial;
};

void pairTerm(const simd::LjParams &params, double r2, double &pair_e, double &fscale) {
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

void applyPair(double *local, int origin, std::int64_t i, std::int64_t j, double dx, double dy,
               double dz, double r2, double cutoff2, const simd::LjParams &params, double &energy) {
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
  std::size_t words = 0;
  for (int c = 0; c < k; ++c) {
    const auto range = nlist::partition(n, c, k);
    scratch.begin[static_cast<std::size_t>(c)] = static_cast<int>(range.begin);
    scratch.end[static_cast<std::size_t>(c)] = static_cast<int>(range.end);
    // A half pair i < j writes +f at i and -f at j, so the buffer starts
    // at the first i of the chunk and runs to the last atom.
    scratch.origin[static_cast<std::size_t>(c)] = static_cast<int>(range.begin);
    const std::size_t span = static_cast<std::size_t>(n - range.begin);
    words += span * 3;
  }
  scratch.slab.assign(words, 0.0);
  int *begin = scratch.begin.data();
  int *end = scratch.end.data();
  int *origin = scratch.origin.data();
  double *slab = scratch.slab.data();
  double *partial = scratch.partial.data();
  std::vector<std::size_t> cursor(static_cast<std::size_t>(k) + 1, 0);
  for (int c = 0; c < k; ++c) {
    cursor[static_cast<std::size_t>(c) + 1] =
        cursor[static_cast<std::size_t>(c)] +
        static_cast<std::size_t>(n - origin[c]) * 3;
  }
  std::size_t *base = cursor.data();
  auto run_chunk = [&](int c) {
      double *local = slab + base[c];
      const int org = origin[c];
      double local_e = 0.0;
      const std::int64_t i0 = begin[c];
      const std::int64_t i1 = end[c];
      if (plain) {
        for (std::int64_t i = i0; i < i1; ++i) {
          const auto ui = static_cast<std::size_t>(i) * 3;
          const double xi = pos[ui];
          const double yi = pos[ui + 1];
          const double zi = pos[ui + 2];
          const auto row = static_cast<std::size_t>(i);
          for (std::int32_t p = off[row]; p < off[row + 1]; ++p) {
            const auto j = static_cast<std::int64_t>(nbr[static_cast<std::size_t>(p)]);
            if (j <= i) {
              continue;
            }
            const auto uj = static_cast<std::size_t>(j) * 3;
            const double dx = xi - pos[uj];
            const double dy = yi - pos[uj + 1];
            const double dz = zi - pos[uj + 2];
            const double r2 = (dx * dx + dy * dy) + dz * dz;
            applyPair(local, org, i, j, dx, dy, dz, r2, cutoff2, params, local_e);
          }
        }
      } else {
        for (std::int64_t i = i0; i < i1; ++i) {
          cells.forDirected(i, i + 1, [&](std::int32_t, std::int32_t j, double dx, double dy,
                                          double dz, double r2) {
            if (static_cast<std::int64_t>(j) <= i) {
              return;
            }
            applyPair(local, org, i, j, dx, dy, dz, r2, cutoff2, params, local_e);
          });
        }
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
