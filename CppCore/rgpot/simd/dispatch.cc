// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/simd/api.hpp"

#if defined(RGPOT_WITH_HIGHWAY)
#include "rgpot/simd/hwy_disabled.h"
#include "hwy/highway.h"
#endif

#include <cstdlib>
#include <cstring>
#include <mutex>

namespace rgpot::simd {
#if defined(RGPOT_WITH_HIGHWAY)
// Bind rgpot's function pointers. Does not call Highway's process-global
// DisableTargets or SetSupportedTargetsForTest. targets == 0 selects the
// best bit in SupportedTargets() masked by the compiled set.
bool bind_hwy_pointers(std::int64_t targets);
std::int64_t active_hwy_bit();
#endif
namespace {

enum class Mode { Auto, Scalar, Named };

std::once_flag once;
Mode mode = Mode::Auto;
bool test_override = false;
const char *runnable[16] = {"scalar"};
int nrunnable = 1;

#if defined(RGPOT_WITH_HIGHWAY)
std::int64_t cpu_targets = 0;

struct NamedBit {
  const char *name;
  std::int64_t bit;
};

const NamedBit kBits[] = {
    {"sse2", HWY_SSE2},     {"sse4", HWY_SSE4},
    {"avx2", HWY_AVX2},     {"avx3", HWY_AVX3},
    {"avx3_zen4", HWY_AVX3_ZEN4},
    {"neon", HWY_NEON},     {"sve", HWY_SVE},
    {"sve2", HWY_SVE2},     {"sve_256", HWY_SVE_256},
    {"rvv", HWY_RVV},       {"emu128", HWY_EMU128},
};

std::int64_t name_bit(const char *name) {
  for (const NamedBit &item : kBits) {
    if (std::strcmp(name, item.name) == 0) {
      return item.bit;
    }
  }
  return 0;
}

const char *bit_name(std::int64_t bit) {
  for (const NamedBit &item : kBits) {
    if (item.bit == bit) {
      return item.name;
    }
  }
  return "scalar";
}

// x86-64 always has SSE2. If the local choice landed on the portable
// kernel, pin SSE2 instead. An explicit scalar request is left alone.
void clamp_x86_off_scalar() {
#if defined(__x86_64__) || defined(_M_X64)
  if (mode == Mode::Scalar) {
    return;
  }
  if (std::strcmp(bit_name(active_hwy_bit()), "scalar") == 0 &&
      (cpu_targets & HWY_SSE2) != 0) {
    mode = Mode::Named;
    if (!bind_hwy_pointers(HWY_SSE2)) {
      mode = Mode::Scalar;
    }
  }
#else
  (void)0;
#endif
}

void fill_runnable() {
  nrunnable = 0;
  runnable[nrunnable++] = "scalar";
  for (const NamedBit &item : kBits) {
    if ((cpu_targets & item.bit) != 0) {
      runnable[nrunnable++] = item.name;
    }
  }
}
#endif

void ensure_body() {
#if defined(RGPOT_WITH_HIGHWAY)
  cpu_targets = hwy::SupportedTargets() & static_cast<std::int64_t>(HWY_TARGETS);
  fill_runnable();
  std::int64_t select = 0;
  if (!test_override) {
    const char *env = std::getenv("RGPOT_SIMD");
    if (env != nullptr && env[0] != '\0') {
      if (std::strcmp(env, "scalar") == 0) {
        mode = Mode::Scalar;
      } else {
        const std::int64_t bit = name_bit(env);
        if (bit != 0 && (cpu_targets & bit) != 0) {
          mode = Mode::Named;
          select = bit;
        }
        // A known name this CPU cannot run leaves Auto, the best target
        // it can. On x86-64 that is at least SSE2.
      }
    }
  }
  if (mode != Mode::Scalar) {
    if (!bind_hwy_pointers(select)) {
      mode = Mode::Scalar;
    }
  }
  clamp_x86_off_scalar();
#else
  mode = Mode::Scalar;
  nrunnable = 1;
  runnable[0] = "scalar";
#endif
}

void ensure() { std::call_once(once, ensure_body); }

} // namespace

double lj_scalar_entry(const PairView &view, const LjParams &params, double *F,
                       double strain[6]);
double morse_scalar_entry(const PairView &view, const MorseParams &params,
                          double *F, double strain[6]);
void eam_scalar_entry(long n, const double *R, const EamAtom *atoms,
                      const double *box, double *F, double *U);
void fma_scalar_entry(const double *a, const double *b, const double *c,
                      double *out, std::size_t n);
void fold_scalar_entry(double *x, double *y, double *z, std::size_t n,
                       double w0, double w1, double w2, double i0, double i1,
                       double i2);

#if defined(RGPOT_WITH_HIGHWAY)
double lj_hwy_entry(const PairView &view, const LjParams &params, double *F,
                    double strain[6]);
double morse_hwy_entry(const PairView &view, const MorseParams &params,
                       double *F, double strain[6]);
void eam_hwy_entry(long n, const double *R, const EamAtom *atoms,
                   const double *box, double *F, double *U);
void fma_hwy_entry(const double *a, const double *b, const double *c,
                   double *out, std::size_t n);
void fold_hwy_entry(double *x, double *y, double *z, std::size_t n, double w0,
                    double w1, double w2, double i0, double i1, double i2);
#endif

const char *chosen_target_name() {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode == Mode::Scalar) {
    return "scalar";
  }
  return bit_name(active_hwy_bit());
#else
  return "scalar";
#endif
}

void set_target_for_test(const char *name) {
  ensure();
  if (name == nullptr || name[0] == '\0') {
    return;
  }
  test_override = true;
  if (std::strcmp(name, "scalar") == 0) {
    mode = Mode::Scalar;
    return;
  }
#if defined(RGPOT_WITH_HIGHWAY)
  const std::int64_t bit = name_bit(name);
  if (bit == 0) {
    return;
  }
  if ((cpu_targets & bit) == 0) {
    mode = Mode::Auto;
    if (!bind_hwy_pointers(0)) {
      mode = Mode::Scalar;
    }
    clamp_x86_off_scalar();
    return;
  }
  mode = Mode::Named;
  if (!bind_hwy_pointers(bit)) {
    mode = Mode::Auto;
    if (!bind_hwy_pointers(0)) {
      mode = Mode::Scalar;
    }
  }
#else
  mode = Mode::Scalar;
#endif
}

int runnable_target_count() {
  ensure();
  return nrunnable;
}

const char *runnable_target_name(int index) {
  ensure();
  if (index < 0 || index >= nrunnable) {
    return "scalar";
  }
  return runnable[index];
}

double lj_force(const PairView &view, const LjParams &params, double *F,
                double strain[6]) {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode != Mode::Scalar) {
    return lj_hwy_entry(view, params, F, strain);
  }
#endif
  return lj_scalar_entry(view, params, F, strain);
}

double morse_force(const PairView &view, const MorseParams &params, double *F,
                   double strain[6]) {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode != Mode::Scalar) {
    return morse_hwy_entry(view, params, F, strain);
  }
#endif
  return morse_scalar_entry(view, params, F, strain);
}

void fma_array(const double *a, const double *b, const double *c, double *out,
               std::size_t n) {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode != Mode::Scalar) {
    fma_hwy_entry(a, b, c, out, n);
    return;
  }
#endif
  fma_scalar_entry(a, b, c, out, n);
}

void eam_force(long n, const double *R, const EamAtom *atoms, const double *box,
               double *F, double *U) {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode != Mode::Scalar) {
    eam_hwy_entry(n, R, atoms, box, F, U);
    return;
  }
#endif
  eam_scalar_entry(n, R, atoms, box, F, U);
}

void fold_orthorhombic(double *x, double *y, double *z, std::size_t n,
                       double w0, double w1, double w2, double i0, double i1,
                       double i2) {
  ensure();
#if defined(RGPOT_WITH_HIGHWAY)
  if (mode != Mode::Scalar) {
    fold_hwy_entry(x, y, z, n, w0, w1, w2, i0, i1, i2);
    return;
  }
#endif
  fold_scalar_entry(x, y, z, n, w0, w1, w2, i0, i1, i2);
}

} // namespace rgpot::simd
