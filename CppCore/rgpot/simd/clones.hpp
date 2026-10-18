#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// RGPOT_SIMD_CLONES compiles a loop once per x86-64 vector width and lets
// the loader pick one from the CPU (GCC and clang function multiversioning,
// resolved through libgcc's or compiler-rt's own cpuid, not glibc's). The
// clones name no FMA target, and the units that use this build with
// contraction off, so every clone runs the same IEEE operations and returns
// the same bits. Elsewhere the loop is compiled once for the baseline.

#if defined(__x86_64__) && defined(__linux__) && defined(__has_attribute)
#if __has_attribute(target_clones)
#define RGPOT_SIMD_CLONES                                                      \
  __attribute__((target_clones("default", "avx2", "avx512f")))
#endif
#endif
#if !defined(RGPOT_SIMD_CLONES)
#define RGPOT_SIMD_CLONES
#endif
