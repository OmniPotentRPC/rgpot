The wheel ISA audit names the `target("fma")` kernels in librgpot (`eam_fma` and the other glibc-gated helpers). Those bodies also contain AVX encodings and run only after glibc reports a usable FMA.
