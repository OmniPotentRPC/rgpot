// MIT License
// Copyright 2023--present rgpot developers
//
// A stand-in for a library built from another revision of the headers:
// ForceOut gains a trailing array, so its size changes while every other
// type stays as it is. The shared object exports the layout stamp the real
// library exports, computed from the changed structure.

// Expands the last declaration of ForceOut to two. Nothing else is read
// before the macro is undefined.
#define has_stress has_stress; double fixture_extra[2]
#include "rgpot/ForceStructs.hpp"
#undef has_stress

#define RGPOT_CXX_LAYOUT_BUILDING
#include "rgpot/abi/CxxLayout.hpp"

static_assert(sizeof(rgpot::ForceOut) > 0);

extern "C" {

rgpot_cxx_layout_stamp_t rgpot_cxx_layout_stamp(void) {
  return rgpot::abi::header_stamp();
}

int32_t rgpot_cxx_layout_compatible(const rgpot_cxx_layout_stamp_t *caller) {
  if (caller == nullptr) {
    return 0;
  }
  const rgpot_cxx_layout_stamp_t mine = rgpot::abi::header_stamp();
  return caller->revision == mine.revision && caller->features == mine.features &&
                 caller->hash == mine.hash
             ? 1
             : 0;
}

} // extern "C"
