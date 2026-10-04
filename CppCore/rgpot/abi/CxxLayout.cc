// MIT License
// Copyright 2023--present rgpot developers

#define RGPOT_CXX_LAYOUT_BUILDING
#include "rgpot/abi/CxxLayout.hpp"

namespace {
// Constant-initialised in this translation unit, so the exported functions
// never call an inline function another shared object may also define.
constexpr rgpot_cxx_layout_stamp_t kStamp = rgpot::abi::header_stamp();
} // namespace

extern "C" {

rgpot_cxx_layout_stamp_t rgpot_cxx_layout_stamp(void) {
  return kStamp;
}

int32_t rgpot_cxx_layout_compatible(const rgpot_cxx_layout_stamp_t *caller) {
  if (caller == nullptr) {
    return 0;
  }
  const rgpot_cxx_layout_stamp_t &mine = kStamp;
  return caller->revision == mine.revision && caller->features == mine.features &&
                 caller->hash == mine.hash
             ? 1
             : 0;
}

} // extern "C"
