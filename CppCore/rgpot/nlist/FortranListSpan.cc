// MIT License
// Copyright 2023--present rgpot developers

// The slot count of the Fortran output buffers. The header layout of
// Builder depends on RGPOT_EAM_AL_OPENMP, so this count has to be read
// in the same translation unit family as the list, not by a caller that
// may have been compiled without that macro.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rgpot/nlist/NeighbourList.hpp"

struct rgpot_nlist_fortran_list {
  rgpot::nlist::Builder builder;
  std::vector<std::int32_t> idx;
  std::vector<std::int32_t> atom;
  std::vector<double> vec;
  std::vector<double> dist;
  std::vector<std::int32_t> row;
};

extern "C" __attribute__((visibility("hidden"))) std::size_t
rgpot_nlist_fortran_slot_count(const rgpot_nlist_fortran_list *list) {
  if (list == nullptr) {
    return 0;
  }
  return list->idx.size();
}
