#include "gupta.h"

#include "detail/gupta_test.h"

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

#if defined(_MSC_VER) && !defined(__clang__)
#define RGPOT_THREAD_LOCAL __declspec(thread)
#else
#define RGPOT_THREAD_LOCAL _Thread_local
#endif

static RGPOT_THREAD_LOCAL double *rho_buf;
static RGPOT_THREAD_LOCAL size_t rho_cap;
static RGPOT_THREAD_LOCAL double *pack_buf;
static RGPOT_THREAD_LOCAL size_t pack_cap;
static RGPOT_THREAD_LOCAL void *(*gupta_alloc_hook)(size_t);

static void *gupta_alloc(size_t bytes) {
  if (gupta_alloc_hook != 0) {
    return gupta_alloc_hook(bytes);
  }
  return malloc(bytes);
}

// Allocate every new buffer before replacing a pointer or a capacity.
// A failed pair allocation leaves the density buffer and its capacity.
static int rgpot_gupta_ensure(size_t n, size_t pairs) {
  double *next_rho = 0;
  double *next_pack = 0;
  const size_t need = pairs * 6;
  if (rho_cap < n) {
    next_rho = (double *)gupta_alloc(n * sizeof(double));
    if (next_rho == 0) {
      return 1;
    }
  }
  if (pack_cap < need) {
    next_pack = (double *)gupta_alloc(need * sizeof(double));
    if (next_pack == 0) {
      free(next_rho);
      return 1;
    }
  }
  if (next_rho != 0) {
    free(rho_buf);
    rho_buf = next_rho;
    rho_cap = n;
  }
  if (next_pack != 0) {
    free(pack_buf);
    pack_buf = next_pack;
    pack_cap = need;
  }
  return 0;
}

// The header undefines this after the declarations. Repeat it so the
// definitions stay hidden in the shared library.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_WIN32) &&           \
    !defined(__CYGWIN__)
#define RGPOT_GUPTA_TEST_HIDDEN __attribute__((visibility("hidden")))
#else
#define RGPOT_GUPTA_TEST_HIDDEN
#endif

RGPOT_GUPTA_TEST_HIDDEN void rgpot_gupta_test_set_alloc(void *(*fn)(size_t)) {
  gupta_alloc_hook = fn;
}

RGPOT_GUPTA_TEST_HIDDEN size_t rgpot_gupta_test_rho_cap(void) {
  return rho_cap;
}

RGPOT_GUPTA_TEST_HIDDEN size_t rgpot_gupta_test_pack_cap(void) {
  return pack_cap;
}

static double *rgpot_gupta_rho(void) { return rho_buf; }

static double *rgpot_gupta_pack(void) { return pack_buf; }

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
