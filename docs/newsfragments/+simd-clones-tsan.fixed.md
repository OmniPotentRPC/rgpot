A ThreadSanitizer build of `librgpot` compiles each `RGPOT_SIMD_CLONES` loop once, for the baseline, so `import rgpot._core` no longer crashes in the loader.
The sanitizer had instrumented the clone resolver.
It called the sanitizer hooks through procedure linkage table slots.
An `RTLD_NOW` load had not bound those slots yet.
Every other build keeps the clones, and every clone returns the same bits.
