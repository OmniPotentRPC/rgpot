#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

/// Copy a computed virial and stress onto a PotentialResult.
///
/// Units are the caller's. `ev_to_caller` converts eV to that energy.
/// `len_to_angstrom` converts the caller's length to Angstrom, so
/// stress_caller = stress_eV/A^3 * ev_to_caller * len_to_angstrom^3.
/// The virial is an energy. A flag of 0 leaves that list absent: that
/// is the not-available state. When both lists are present,
/// stress = -virial/V in the caller's units.
inline void writeMechanical(::PotentialResult::Builder pres,
                            const double stress[9], int has_stress,
                            const double virial[9], int has_virial,
                            double ev_to_caller, double len_to_angstrom) {
  const double stress_to_caller =
      ev_to_caller * len_to_angstrom * len_to_angstrom * len_to_angstrom;
  if (has_stress != 0) {
    auto list = pres.initStress(9);
    for (int i = 0; i < 9; ++i) {
      list.set(static_cast<unsigned int>(i), stress[i] * stress_to_caller);
    }
    pres.setHasStress(true);
  }
  if (has_virial != 0) {
    auto list = pres.initVirial(9);
    for (int i = 0; i < 9; ++i) {
      list.set(static_cast<unsigned int>(i), virial[i] * ev_to_caller);
    }
    pres.setHasVirial(true);
  }
}

} // namespace rgpot::rpc
