// MIT License
// Copyright 2023--present rgpot developers
//
// Prints the SHA-256 of folded displacements, or "skip" when this binary
// was built for an instruction set the CPU cannot run.

#include "minimage_hash_samples.hpp"

#include <iostream>

namespace {

bool cpu_can_run() {
#if defined(RGPOT_MINIMAGE_ISA_AVX512)
  return __builtin_cpu_supports("avx512f");
#elif defined(RGPOT_MINIMAGE_ISA_AVX2)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return true;
#endif
}

} // namespace

int main() {
  if (!cpu_can_run()) {
    std::cout << "skip\n";
    return 0;
  }
  std::cout << rgpot::nlist::hashcheck::folded_displacement_hash() << '\n';
  return 0;
}
