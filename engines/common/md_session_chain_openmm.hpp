#pragma once
// MIT License
// Copyright 2023--present rgpot developers

// The harmonic chain of md_session_analytic.hpp as a serialized OpenMM System.

#include "engines/common/md_session_analytic.hpp"
#include "rgpot/ExternalPot/md_units.hpp"

#include <openmm/CustomCompoundBondForce.h>
#include <openmm/System.h>
#include <openmm/serialization/XmlSerializer.h>

#include <sstream>
#include <string>
#include <vector>

namespace rgpot::analytic {

inline std::string chainOpenMMXml(const Chain &chain) {
  OpenMM::System system;
  for (const double mass : chain.masses) {
    system.addParticle(mass);
  }
  // eV / Angstrom^2 to kJ/mol / nm^2.
  const double toNative = md::kKjPerEv * 100.0;
  auto *bond = new OpenMM::CustomCompoundBondForce(
      2, "0.5*k*((x1-x2)^2+(y1-y2)^2+(z1-z2)^2)");
  bond->addPerBondParameter("k");
  bond->addBond({0, 1}, {chain.k1 * toNative});
  bond->addBond({1, 2}, {chain.k2 * toNative});
  system.addForce(bond);
  std::ostringstream xml;
  OpenMM::XmlSerializer::serialize<OpenMM::System>(&system, "System", xml);
  return xml.str();
}

} // namespace rgpot::analytic
