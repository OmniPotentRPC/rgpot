// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/fortran/FortranPots.hpp"

namespace rgpot::fortranpots {

bool openmpGather() noexcept {
#ifdef RGPOT_EAM_AL_OPENMP
  return true;
#else
  return false;
#endif
}

} // namespace rgpot::fortranpots
