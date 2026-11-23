#pragma once
// MIT License
// Copyright 2026 Rohit Goswami

// The harmonic chain of md_session_analytic.hpp as an nblib topology: three
// atom types of the chain masses, no nonbonded parameters, and two harmonic
// bonds of rest length zero.

#include "engines/common/md_session_analytic.hpp"
#include "rgpot/ExternalPot/md_units.hpp"

#include <limits>
#include <sstream>
#include <string>

namespace rgpot::analytic {

inline std::string chainNblibTopology(const Chain &chain) {
  // eV / Angstrom^2 to kJ/mol / nm^2.
  const double toNative = md::kKjPerEv * 100.0;
  std::ostringstream text;
  text.precision(std::numeric_limits<double>::max_digits10);
  text << "nblib 1\ncutoff 0.8\ncoulomb none\n"
       << "atomtype A " << chain.masses[0] << " 0 0\n"
       << "atomtype B " << chain.masses[1] << " 0 0\n"
       << "atomtype C " << chain.masses[2] << " 0 0\n"
       << "molecule chain 1\natom A1 A 0\natom B1 B 0\natom C1 C 0\n"
       << "bond harmonic A1 B1 " << chain.k1 * toNative << " 0\n"
       << "bond harmonic B1 C1 " << chain.k2 * toNative << " 0\nend\n";
  return text.str();
}

} // namespace rgpot::analytic
