// MIT License
// Copyright 2023--present rgpot developers

// RGPOT_SIMD is read once, so this driver is a separate process from the
// Catch2 suite.

#include "rgpot/simd/api.hpp"

#include <cstring>
#include <iostream>

namespace {

bool runnable(const char *name) {
  const int n = rgpot::simd::runnable_target_count();
  for (int i = 0; i < n; ++i) {
    if (std::strcmp(rgpot::simd::runnable_target_name(i), name) == 0) {
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "print";
  const char *got = rgpot::simd::chosen_target_name();
  if (std::strcmp(mode, "print") == 0) {
    std::cout << got << "\n";
    return 0;
  }
  if (std::strcmp(mode, "unknown") == 0) {
    bool found = false;
    const int n = rgpot::simd::runnable_target_count();
    for (int i = 0; i < n; ++i) {
      if (std::strcmp(rgpot::simd::runnable_target_name(i), got) == 0) {
        found = true;
      }
    }
    if (!found) {
      std::cerr << "chosen target is not runnable: " << got << "\n";
      return 1;
    }
    if (n > 1 && std::strcmp(got, "scalar") == 0) {
      std::cerr << "unknown RGPOT_SIMD forced scalar\n";
      return 1;
    }
    return 0;
  }
  if (std::strcmp(mode, "known") == 0) {
    if (argc < 3) {
      return 2;
    }
    const char *want = argv[2];
    if (std::strcmp(want, "scalar") == 0 || !runnable(want)) {
      if (std::strcmp(got, "scalar") != 0) {
        std::cerr << "expected scalar, got " << got << "\n";
        return 1;
      }
      return 0;
    }
    if (std::strcmp(got, want) != 0) {
      std::cerr << "expected " << want << ", got " << got << "\n";
      return 1;
    }
    return 0;
  }
  std::cerr << "usage: simd_env print|unknown|known <name>\n";
  return 2;
}
