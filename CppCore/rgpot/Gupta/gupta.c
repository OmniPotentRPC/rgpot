#include "gupta.h"

#include <math.h>
#include <stdlib.h>

// The AVX2 clone and its CPU probe are x86 only. Other targets build the
// scalar kernel alone.
// The clone stays static and reached only behind have_avx2_fma(): the ISA
// audit accepts its AVX only after proving that from the library.
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define RGPOT_AVX2_CLONE 1
#define RGPOT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#endif

// Small clusters use stack scratch; larger calls cap stored pair derivatives
// at 32 MiB and free the heap scratch before returning, recomputing pairs
// past the cap identically in the force pass.
//
// Stack scratch covers every cluster of up to RGPOT_GUPTA_STACK_ATOMS atoms,
// so it holds one pair record for each of the N (N - 1) / 2 pairs of the
// largest such cluster. Fewer pairs would send the upper atom counts to malloc.
#define RGPOT_GUPTA_STACK_ATOMS 64
#define RGPOT_GUPTA_STACK_PAIRS \
  (RGPOT_GUPTA_STACK_ATOMS * (RGPOT_GUPTA_STACK_ATOMS - 1) / 2)

// One stored pair is RGPOT_GUPTA_PACK_STRIDE doubles: the separation
// (dx, dy, dz) and its length r, then the derivatives d(phi)/dr of the
// repulsive term and d(psi)/dr of the density term.
#define RGPOT_GUPTA_PACK_STRIDE 6
#define RGPOT_GUPTA_PACK_DX 0
#define RGPOT_GUPTA_PACK_DY 1
#define RGPOT_GUPTA_PACK_DZ 2
#define RGPOT_GUPTA_PACK_R 3
#define RGPOT_GUPTA_PACK_DPHI 4
#define RGPOT_GUPTA_PACK_DPSI 5

// Byte budget for stored pairs. It is the 32 MiB that PairTerms::kDefaultLimit
// in rgpot/PairTerms.hpp keeps for the C++ potentials, repeated here because
// this file is C.
#define RGPOT_GUPTA_PACK_BUDGET_BYTES (32u << 20)
#define RGPOT_GUPTA_MAX_PACK_PAIRS \
  (RGPOT_GUPTA_PACK_BUDGET_BYTES /  \
   (RGPOT_GUPTA_PACK_STRIDE * sizeof(double)))

static void gupta_scalar(const double *x, size_t n, const unsigned char *slot,
                         size_t n_species, const double *a, const double *xi, const double *p,
                         const double *q, const double *r0, double *energy, double *gradient) {
#include "gupta_body.inc"
}

#ifdef RGPOT_AVX2_CLONE
RGPOT_TARGET_AVX2
static void gupta_avx2(const double *x, size_t n, const unsigned char *slot, size_t n_species,
                       const double *a, const double *xi, const double *p, const double *q,
                       const double *r0, double *energy, double *gradient) {
#include "gupta_body.inc"
}

static int have_avx2_fma(void) {
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}
#endif

void rgpot_gupta(const double *x, size_t n, const unsigned char *slot, size_t n_species,
                 const double *a, const double *xi, const double *p, const double *q,
                 const double *r0, double *energy, double *gradient) {
#ifdef RGPOT_AVX2_CLONE
  if (have_avx2_fma()) {
    gupta_avx2(x, n, slot, n_species, a, xi, p, q, r0, energy, gradient);
    return;
  }
#endif
  gupta_scalar(x, n, slot, n_species, a, xi, p, q, r0, energy, gradient);
}
