/* MIT License
 * Copyright 2023--present rgpot developers
 *
 * fma() from mingw-w64's libmingwex returns wrong results on x86-64.
 *
 *   x86_64-w64-mingw32-gcc -O2 -fno-builtin -o fma_repro.exe fma_repro.c
 *   wine ./fma_repro.exe        (or run it on Windows)
 *
 * Prints one line per case and exits with the number of mismatches.
 * Build it natively (cc -O2 -fno-builtin fma_repro.c -lm) for a reference
 * run against glibc. Define FMA=name to test another implementation.
 */
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef FMA
#define FMA fma
#endif
double FMA(double, double, double);

static double from_bits(uint64_t u) {
  double d;
  memcpy(&d, &u, sizeof d);
  return d;
}

static uint64_t to_bits(double d) {
  uint64_t u;
  memcpy(&u, &d, sizeof u);
  return u;
}

struct fma_case {
  const char *what;
  uint64_t x, y, z, want;
};

static const struct fma_case cases[] = {
    /* 1 * (1 + 2^-52) + (1 + 2^-52) = 2 + 2^-51, exactly representable. */
    {"exact result", 0x3ff0000000000000, 0x3ff0000000000001, 0x3ff0000000000001,
     0x4000000000000001},
    /* (1 + 2^-52)^2 + pi: the exact sum lies just above a rounding tie, so
       it rounds up. Rounding x*y first lands on the tie, which rounds to
       even (down). */
    {"tie broken by low product bits", 0x3ff0000000000001, 0x3ff0000000000001,
     0x400921fb54442d18, 0x401090fdaa22168d},
    /* 0 * inf + qNaN: the result is a NaN either way; IEEE 754-2019 6.2.3
       says it should be the input NaN, as x86 FMA3 and glibc return. */
    {"0 * inf + qNaN", 0x0000000000000000, 0x7ff0000000000000,
     0x7ff8000000000000, 0x7ff8000000000000},
};

int main(void) {
  int bad = 0;
  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
    const struct fma_case *c = &cases[i];
    /* volatile keeps the compiler from folding fma at compile time. */
    volatile double x = from_bits(c->x), y = from_bits(c->y),
                    z = from_bits(c->z);
    uint64_t got = to_bits(FMA(x, y, z));
    int ok = got == c->want;
    bad += !ok;
    printf("case %zu (%s): fma(%016" PRIx64 ", %016" PRIx64 ", %016" PRIx64
           ") want %016" PRIx64 " got %016" PRIx64 " %s\n",
           i + 1, c->what, c->x, c->y, c->z, c->want, got, ok ? "ok" : "WRONG");
  }
  return bad;
}
