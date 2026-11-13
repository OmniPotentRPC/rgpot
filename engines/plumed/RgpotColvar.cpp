// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright 2023--present rgpot developers
//
// PLUMED collective variable RGPOT. This file includes the PLUMED
// headers, so it is distributed under the PLUMED license. The rest of
// rgpot stays MIT.
//
// Parameters and the reported value are in the units PLUMED is using.
// GROMACS leaves the default (nm, kJ/mol). A driver that sets
// UNITS LENGTH=A ENERGY=eV passes Angstrom and eV straight through.
// The value is the shifted Lennard-Jones energy. Its atomic derivatives
// are dE/dx, so BIASVALUE applies the rgpot forces.

#include "plumed/colvar/Colvar.h"
#include "plumed/core/ActionRegister.h"
#include "plumed/core/Atoms.h"
#include "plumed/core/PlumedMain.h"
#include "plumed/tools/Units.h"

#include "lj_eval.hpp"
#include "rgpot/units.hpp"

#include <vector>

namespace PLMD {
namespace colvar {

class Rgpot : public Colvar {
 public:
  static void registerKeywords(Keywords &keys);
  explicit Rgpot(const ActionOptions &ao);
  void calculate() override;

 private:
  double u0_ = 1.0;
  double cutoff_ = 8.0;
  double psi_ = 1.0;
  bool nopbc_ = false;
};

PLUMED_REGISTER_ACTION(Rgpot, "RGPOT")

void Rgpot::registerKeywords(Keywords &keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms", "ATOMS", "Atoms of the rgpot evaluation");
  keys.add("compulsory", "U0", "1.0", "Well depth, in the current energy unit");
  keys.add("compulsory", "CUTOFF", "8.0", "Cutoff, in the current length unit");
  keys.add("compulsory", "PSI", "1.0", "Zero crossing, in the current length unit");
}

Rgpot::Rgpot(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao) {
  std::vector<AtomNumber> atoms;
  parseAtomList("ATOMS", atoms);
  parse("U0", u0_);
  parse("CUTOFF", cutoff_);
  parse("PSI", psi_);
  parseFlag("NOPBC", nopbc_);
  checkRead();
  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
}

void Rgpot::calculate() {
  const Units &units = plumed.getAtoms().getUnits();
  const double to_angstrom = units.getLength() / 0.1;
  const double to_ev = units.getEnergy() * rgpot::units::unit_conversion_factor("kJ/mol", "eV");
  const auto &positions = getPositions();
  const unsigned n = static_cast<unsigned>(positions.size());
  std::vector<double> xyz(static_cast<size_t>(n) * 3);
  std::vector<int> atomic_numbers(n, 1);
  for (unsigned i = 0; i < n; ++i) {
    xyz[3 * i] = positions[i][0] * to_angstrom;
    xyz[3 * i + 1] = positions[i][1] * to_angstrom;
    xyz[3 * i + 2] = positions[i][2] * to_angstrom;
  }
  double box[9] = {20, 0, 0, 0, 20, 0, 0, 0, 20};
  if (!nopbc_) {
    const Tensor &cell = getBox();
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        box[3 * row + col] = cell(row, col) * to_angstrom;
      }
    }
  }
  std::vector<double> forces(static_cast<size_t>(n) * 3);
  double energy_ev = 0.0;
  int has_stress = 0;
  double stress[9];
  if (rgpot_lj_eval(xyz.data(), atomic_numbers.data(), n, box, u0_ * to_ev,
                    cutoff_ * to_angstrom, psi_ * to_angstrom, forces.data(), &energy_ev,
                    &has_stress, stress) != 0) {
    error("RGPOT could not evaluate this geometry");
  }
  const double energy = energy_ev / to_ev;
  setValue(energy);
  const double force_scale = to_angstrom / to_ev;
  for (unsigned i = 0; i < n; ++i) {
    // dE/dx = -F, in PLUMED's units.
    setAtomsDerivatives(static_cast<int>(i),
                        Vector(-forces[3 * i] * force_scale, -forces[3 * i + 1] * force_scale,
                               -forces[3 * i + 2] * force_scale));
  }
  setBoxDerivativesNoPbc();
}

} // namespace colvar
} // namespace PLMD
