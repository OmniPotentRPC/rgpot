The tight Lennard-Jones cluster kernel's AVX2 bodies run only after
`__builtin_cpu_supports("avx2")`.
`rgpot/isa-allow.toml` names `lj_avx2`, `eval_quad`, and `acc_quad`
for AVX and AVX2, and names GCC's `__cpu_indicator_init` for the
`xgetbv` that initializer uses.
