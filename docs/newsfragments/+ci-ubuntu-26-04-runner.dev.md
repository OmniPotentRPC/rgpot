The shared `py-tsan` workflow sets `vm.mmap_rnd_bits=28` unconditionally
and runs its container with `seccomp=unconfined`, and the ISA self-test
compiles its x86 fixtures at `-march=x86-64`. Both follow the
`ubuntu-latest` runner's move to Ubuntu 26.04, where the sysctl is
root-only to read and the default compiler emits VEX scalar moves.
