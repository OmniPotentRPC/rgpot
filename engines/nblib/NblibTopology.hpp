#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief The topology a nblib potential or session is built from.
 *
 * Plain data in the units of a GROMACS force field: nm, kJ/mol, amu and
 * elementary charge. The text format is described in
 * docs/orgmode/howto/engine-nblib.org. Nothing here includes nblib, so the
 * parser builds and runs without a GROMACS installation.
 */

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rgpot::nblibpot {

enum class CoulombMode { None, Cutoff };

struct AtomTypeDesc {
  std::string name;
  double mass = 0.0;  //!< amu
  double c6 = 0.0;    //!< kJ/mol nm^6
  double c12 = 0.0;   //!< kJ/mol nm^12
};

/// Explicit C6 and C12 of one pair of types, replacing the geometric rule.
struct PairDesc {
  std::string a;
  std::string b;
  double c6 = 0.0;
  double c12 = 0.0;
};

struct AtomDesc {
  std::string name;
  std::string type;
  double charge = 0.0;
};

enum class ListedKind {
  HarmonicBond,   //!< k (kJ/mol nm^2), r0 (nm)
  G96Bond,        //!< k (kJ/mol nm^4), r0 (nm)
  MorseBond,      //!< depth (kJ/mol), beta (1/nm), r0 (nm)
  Pair14,         //!< c6, c12 of a bonded pair, with no exclusion implied
  HarmonicAngle,  //!< k (kJ/mol rad^2), theta0 (degrees)
  G96Angle,       //!< k (kJ/mol), theta0 (degrees)
  ProperDihedral, //!< phi0 (degrees), k (kJ/mol), multiplicity
  ImproperDihedral, //!< phi0 (degrees), k (kJ/mol rad^2)
  RyckaertBelleman  //!< c0 .. c5 (kJ/mol)
};

/// Number of atoms and parameters a kind takes.
std::size_t centres(ListedKind kind);
std::size_t parameterCount(ListedKind kind);

struct ListedDesc {
  ListedKind kind = ListedKind::HarmonicBond;
  std::vector<int> atoms; //!< indices into the molecule's atoms
  std::vector<double> params;
};

struct MoleculeDesc {
  std::string name;
  int count = 1;
  std::vector<AtomDesc> atoms;
  std::vector<std::array<int, 2>> exclusions;
  std::vector<ListedDesc> listed;
};

struct TopologyDesc {
  double cutoff = 1.0; //!< nm; pair list cutoff and interaction cutoff
  CoulombMode coulomb = CoulombMode::None;
  std::vector<AtomTypeDesc> types;
  std::vector<PairDesc> pairs;
  std::vector<MoleculeDesc> molecules;

  [[nodiscard]] std::size_t atomCount() const;
  /// Type of every atom in system order.
  [[nodiscard]] std::vector<const AtomTypeDesc *> atomTypes() const;
  /// Charge of every atom in system order, zero when coulomb is none.
  [[nodiscard]] std::vector<double> charges() const;
  /// One entry per molecule instance: the first atom index and the size.
  [[nodiscard]] std::vector<std::array<std::size_t, 2>> instances() const;
};

/// Parses and checks the text. Throws std::invalid_argument with the line.
TopologyDesc parseTopology(std::string_view text);

} // namespace rgpot::nblibpot
