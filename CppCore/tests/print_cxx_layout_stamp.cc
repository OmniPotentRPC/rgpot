// MIT License
// Copyright 2023--present rgpot developers
//
// Prints the no-cache layout stamp for this compiler. CI copies the line
// into kCxxSoversionTable. Does not link librgpot.

#include <iomanip>
#include <iostream>

#include "rgpot/abi/CxxLayout.hpp"

int main() {
  constexpr auto stamp = rgpot::abi::header_stamp();
  std::cout << "RGPOT_CXX_LAYOUT_STAMP"
            << " tag=" << rgpot::abi::kCxxPlatformTag << " hash=";
  std::cout << std::hex << std::setfill('0') << std::setw(16) << stamp.hash;
  std::cout << std::dec << " hash_dec=" << stamp.hash
            << " features=" << stamp.features
            << " revision=" << stamp.revision << '\n';
  return stamp.hash == 0 ? 1 : 0;
}
