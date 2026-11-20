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
//
// ThreadSanitizer instruments the resolver, which then calls
// __tsan_func_entry through the PLT. An RTLD_NOW load runs the resolver
// while it binds the PLT in order, so a slot that the resolver needs may
// still be unbound, and the import jumps to an unrelocated address. A TSan
// build therefore takes the baseline copy, which returns the same bits.

#if defined(__SANITIZE_THREAD__)
#define RGPOT_SIMD_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define RGPOT_SIMD_TSAN 1
#endif
#endif

#if defined(__x86_64__) && defined(__linux__) && defined(__has_attribute) &&   \
    !defined(RGPOT_SIMD_TSAN)
#if __has_attribute(target_clones)
#define RGPOT_SIMD_CLONES                                                      \
  __attribute__((target_clones("default", "avx2", "avx512f")))
#endif
#endif
#if !defined(RGPOT_SIMD_CLONES)
#define RGPOT_SIMD_CLONES
#endif
