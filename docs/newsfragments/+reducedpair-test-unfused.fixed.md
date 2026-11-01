The Meson build compiles `ReducedPairTest` with `-ffp-contract=off`, as it already compiles the `ReducedPair` library.
Apple Clang had fused the expected Morse curvature in the test into an `fma`, which missed the library value by one ulp.
