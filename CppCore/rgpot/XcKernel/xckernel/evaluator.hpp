#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Wraps the generated libxckernel evaluator.hpp and replaces its double
// precision stage B with BLAS-3. The generated header stays untouched, so
// regenerating third_party/libxckernel keeps this in force: the kernels
// include "xckernel/evaluator.hpp", this directory precedes the generated
// include directory, and the relative include below reaches the generated
// header.
//
// Stage B is out(u,v) += sum_g U(u,g) c(g) V(v,g), a matrix product of the
// weighted block (U * c) with V^T. The grid is processed in blocks; each
// block is one dgemm, or one dsyr2k when U and V are the same array (the
// Fock-like blocks, symmetric by construction).

#include "../../../../third_party/libxckernel/include/xckernel/evaluator.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

// Fortran BLAS interface (LP64), resolved by FlexiBLAS or the generic blas
// the build links. Character arguments are single letters.
extern "C" {
void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
            const int *k, const double *alpha, const double *a, const int *lda,
            const double *b, const int *ldb, const double *beta, double *c,
            const int *ldc);
void dsyr2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const double *alpha, const double *a, const int *lda,
             const double *b, const int *ldb, const double *beta, double *c,
             const int *ldc);
}

namespace xckernel {

/// Calls of the BLAS stage_b in this process. The kernels increment it, so a
/// test that reads it proves the kernels resolve to this wrapper and not to
/// the generated scalar loop.
inline std::atomic<unsigned long> &blas_stage_b_calls() {
  static std::atomic<unsigned long> n{0};
  return n;
}

// Stage A forms the per-point coefficient, a sum of monomials of a few
// factors. It accumulates in long double and rounds once, which keeps the
// coefficient (and through it the Fock and fxc matrices) within a rounding of
// the NumPy evaluation.
template <>
XCK_HD inline void stage_a<double, double>(
    int64_t npts, int64_t nm, const double *cf, const int32_t *off,
    const uint16_t *fid, int64_t nfld, const double *const *fields,
    const double *const *xc, double *c) {
  for (int64_t g = 0; g < npts; ++g) {
    long double acc = 0.0L;
    for (int64_t m = 0; m < nm; ++m) {
      long double t = static_cast<long double>(cf[m]);
      for (int32_t f = off[m]; f < off[m + 1]; ++f) {
        const uint16_t id = fid[f];
        t *= (id < nfld) ? fields[id][g] : xc[id - nfld][g];
      }
      acc += t;
    }
    c[g] = static_cast<double>(acc);
  }
}

template <>
inline void stage_b<double>(int64_t npts, int64_t nbf, const double *U,
                            const double *c, const double *V, double *out) {
  blas_stage_b_calls().fetch_add(1, std::memory_order_relaxed);
  if (npts <= 0 || nbf <= 0) {
    return;
  }
  // Each block of kGridBlock points is one dgemm (or dsyr2k), so the double
  // rounding of a block sum grows with kGridBlock and not with npts. The
  // block results are added in long double and rounded once, which keeps the
  // total within a rounding of a long-double sum over the whole grid.
  constexpr int64_t kGridBlock = 256;
  const int64_t blk = std::min<int64_t>(npts, kGridBlock);
  const std::size_t n2 = static_cast<std::size_t>(nbf * nbf);
  std::vector<double> uc(static_cast<std::size_t>(nbf * blk));
  std::vector<double> part(n2);
  std::vector<long double> acc(n2, 0.0L);
  const bool symmetric = (U == V);
  const int n = static_cast<int>(nbf);
  const double one = 1.0;
  const double zero = 0.0;
  const double half = 0.5;
  for (int64_t g0 = 0; g0 < npts; g0 += blk) {
    const int k = static_cast<int>(std::min(blk, npts - g0));
    for (int64_t u = 0; u < nbf; ++u) {
      const double *Ug = U + u * npts + g0;
      double *w = uc.data() + u * k;
      for (int g = 0; g < k; ++g) {
        w[g] = Ug[g] * c[g0 + g];
      }
    }
    const int ldv = static_cast<int>(npts);
    // Column-major views: uc is k x nbf (ld k) and V is k x nbf (ld npts).
    // The row-major out(u,v) is the column-major (v,u) element, so
    // part = V-view^T * uc-view holds out^T.
    if (symmetric) {
      dsyr2k_("U", "T", &n, &k, &half, uc.data(), &k, V + g0, &ldv, &zero,
              part.data(), &n);
      // Only the upper triangle is written; mirror it.
      for (int64_t u = 0; u < nbf; ++u) {
        for (int64_t v = u + 1; v < nbf; ++v) {
          part[static_cast<std::size_t>(u * nbf + v)] =
              part[static_cast<std::size_t>(v * nbf + u)];
        }
      }
    } else {
      dgemm_("T", "N", &n, &n, &k, &one, V + g0, &ldv, uc.data(), &k, &zero,
             part.data(), &n);
    }
    for (std::size_t i = 0; i < n2; ++i) {
      acc[i] += part[i];
    }
  }
  // The column-major (v,u) element and the row-major (u,v) element share an
  // address, so the accumulated block sums add straight into out.
  for (std::size_t i = 0; i < n2; ++i) {
    out[i] += static_cast<double>(acc[i]);
  }
}

} // namespace xckernel
