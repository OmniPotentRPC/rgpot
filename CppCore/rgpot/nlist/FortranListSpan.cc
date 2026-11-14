// MIT License
// Copyright 2023--present rgpot developers

// The slot count of the Fortran output buffers. Read here so a test can
// see the stored length, which is longer than the published pair count.

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
