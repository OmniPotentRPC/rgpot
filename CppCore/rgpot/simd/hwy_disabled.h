#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// Highway 1.2 target bits that this tree does not compile.
// Included before any Highway header. The same literal is the wrap's
// -DHWY_DISABLED_TARGETS, so libhwy and the kernel TUs agree.
//
// Compiled: SSE2 (14), SSE4 (11), AVX2 (9), AVX3 (8), AVX3_ZEN4 (6),
// NEON (28), SVE (24), SVE2 (23), SVE_256 (19), RVV (37), EMU128 (61),
// SCALAR (62).
//
// Not compiled: AVX3_SPR (4), AVX3_DL (7), SSSE3 (12), SVE2_128 (18),
// NEON_BF16 (26), NEON_WITHOUT_AES (29), PPC10/9/8 (47/48/49),
// Z15/Z14 (50/51), WASM_EMU256 (58), WASM (59).
// Highway 1.2 reserves bit 5 for AVX10.2 and has no such target.
// AVX3_DL, AVX3_SPR and AVX10.2 add nothing measured for these f64
// pair kernels; do not enable one without a benchmark that shows a win.
#ifndef HWY_DISABLED_TARGETS
#define HWY_DISABLED_TARGETS (0xC0F800024041090LL)
#endif
