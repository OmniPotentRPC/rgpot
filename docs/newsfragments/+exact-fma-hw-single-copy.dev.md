`librgpot` on x86-64 now carries one copy of `exact_fma_hw`, the one in `exp.cc`.
The weak copy in `exact_fma_hw.cc` stayed in `.text` with no symbol, and the instruction set audit of the Linux x86-64 wheels failed on its `vfmadd132sd`.
