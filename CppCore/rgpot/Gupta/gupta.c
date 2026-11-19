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
#define RGPOT_GUPTA_STACK_ATOMS 64
#define RGPOT_GUPTA_STACK_PAIRS \
  (RGPOT_GUPTA_STACK_ATOMS * (RGPOT_GUPTA_STACK_ATOMS - 1) / 2)
#define RGPOT_GUPTA_MAX_PACK_PAIRS ((32u << 20) / (6u * sizeof(double)))

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
