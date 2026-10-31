The minimum-image fold compiles for Linux aarch64 with GCC.
Its FMA clone keeps the `target("fma")` attribute on x86 only, since aarch64 GCC rejects it and FMA is part of armv8-a.
