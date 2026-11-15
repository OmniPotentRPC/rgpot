// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright 2023--present rgpot developers
//
// PLUMED action RGPOT. This file includes the PLUMED headers, so it is
// distributed under the PLUMED license. The rest of rgpot stays MIT.
//
//   ene: RGPOT POTENTIAL=Name [OPTIONS=key=value,...] ATOMS=... ELEMENTS=El,...
//   ene: RGPOT POTENTIAL=Name SPECIES1=... SPECIES2=... ELEMENTS=El1,El2
//   ene: RGPOT ATOMS=... [U0=u0] [PSI=psi] [CUTOFF=rc]
//
// POTENTIAL is a name rgpot::spec::build takes (pot_spec.cpp), with its
// OPTIONS in rgpot units, Angstrom and eV, as for pair_coeff * * Name.
// ELEMENTS holds one symbol per atom of ATOMS, one per SPECIESn group, or
// one for all atoms. CUTOFF, U0 and PSI are in PLUMED's units; without
// POTENTIAL they are the shifted Lennard-Jones well, zero crossing and
// cutoff, with defaults 1, 1 and 8.
//
// The value is the potential energy in PLUMED's energy unit. Its atomic
// derivatives are dE/dx, so BIASVALUE ARG=ene applies the rgpot forces.
// The cell is PLUMED's periodic box. NOPBC instead puts the atoms in an
// orthorhombic cell that leaves VACUUM between periodic images, which
// must exceed the potential's cutoff; the box derivatives then follow from
// the positions. With a periodic box they are -V sigma, from the
// potential's Cauchy stress (VIRIAL=AUTO) or, when it has none or with
// VIRIAL=FD, from central differences over six strains of the cell, twelve
// more evaluations per step.
//
// Rank 0 of PLUMED's communicator evaluates and broadcasts, so a domain
// decomposed run sees the result of the serial one.

#include "plumed/colvar/Colvar.h"
#include "plumed/config/version.h"
#include "plumed/core/ActionRegister.h"
#include "plumed/core/PlumedMain.h"
#include "plumed/tools/Communicator.h"
#include "plumed/tools/Units.h"
#if PLUMED_VERSION_MAJOR == 2 && PLUMED_VERSION_MINOR < 10
#include "plumed/core/Atoms.h"
#endif

#include "pot_spec.hpp"
#include "rgpot/ExternalPot/external_potentials.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace PLMD {
namespace colvar {

namespace {

constexpr double kStrain = 1.0e-5;

const Units &plumedUnits(PlumedMain &plumed) {
#if PLUMED_VERSION_MAJOR == 2 && PLUMED_VERSION_MINOR < 10
  return plumed.getAtoms().getUnits();
#else
  return plumed.getUnits();
#endif
}

std::string number(double x) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.17g", x);
  return buf;
}

} // namespace

class Rgpot : public Colvar {
 public:
  static void registerKeywords(Keywords &keys);
  explicit Rgpot(const ActionOptions &ao);
  void calculate() override;

 private:
  rgpot::spec::Built built_;
  bool stress_ = false;
  bool virial_fd_ = false;
  bool nopbc_ = false;
  double vacuum_ = 0.0;
  // Angstrom per PLUMED length, PLUMED energy per eV.
  double to_angstrom_ = 1.0;
  double energy_per_ev_ = 1.0;
  std::vector<int> z_;

  int evaluate(const std::vector<double> &xyz, const double *box, double *energy,
               double *forces, double *stress, int *has_stress, std::string &why) const;
  int strainVirial(const std::vector<double> &xyz, const double *box, double w[9],
                   std::string &why) const;
};

PLUMED_REGISTER_ACTION(Rgpot, "RGPOT")

void Rgpot::registerKeywords(Keywords &keys) {
  Colvar::registerKeywords(keys);
  keys.add("atoms", "ATOMS", "Atoms of the rgpot evaluation");
  keys.add("numbered", "SPECIES",
           "Atoms of one species; SPECIES1, SPECIES2, ... take the ELEMENTS in order");
  keys.reset_style("SPECIES", "atoms");
  keys.add("optional", "POTENTIAL",
           "rgpot potential, a name pair_coeff * * Name takes (LJ, Morse, SuttonChen, "
           "EAMTable, ...). Without it the action is the shifted Lennard-Jones potential "
           "of U0, PSI and CUTOFF");
  keys.add("optional", "OPTIONS",
           "Options of POTENTIAL as key=value pairs, in Angstrom and eV");
  keys.add("optional", "ELEMENTS",
           "Element symbols: one per atom of ATOMS, one per SPECIESn group, or one for all");
  keys.add("optional", "CUTOFF",
           "Cutoff in PLUMED's length unit, for a potential whose OPTIONS name none. "
           "Default 8 without POTENTIAL, otherwise the potential's own");
  keys.add("optional", "U0", "Well depth in PLUMED's energy unit, without POTENTIAL. Default 1");
  keys.add("optional", "PSI",
           "Zero crossing in PLUMED's length unit, without POTENTIAL. Default 1");
  keys.add("optional", "VACUUM",
           "With NOPBC, the empty space between periodic images, in PLUMED's length unit. "
           "It must exceed the potential's cutoff; default CUTOFF when that is set");
  keys.add("compulsory", "VIRIAL", "AUTO",
           "AUTO takes the box derivatives from the potential's stress when it has one, FD "
           "always from finite strains of the cell");
}

Rgpot::Rgpot(const ActionOptions &ao) : PLUMED_COLVAR_INIT(ao) {
  const Units &units = plumedUnits(plumed);
  to_angstrom_ = units.getLength() / rgpot_md_nm_per_angstrom();
  energy_per_ev_ = rgpot_md_kj_per_ev() / units.getEnergy();

  std::vector<AtomNumber> atoms;
  std::vector<std::size_t> group_size;
  parseAtomList("ATOMS", atoms);
  if (!atoms.empty()) {
    group_size.push_back(atoms.size());
  }
  for (int k = 1;; ++k) {
    std::vector<AtomNumber> group;
    parseAtomList("SPECIES", k, group);
    if (group.empty()) {
      break;
    }
    if (!group_size.empty() && k == 1) {
      error("give ATOMS or SPECIES1, SPECIES2, ..., not both");
    }
    group_size.push_back(group.size());
    atoms.insert(atoms.end(), group.begin(), group.end());
  }
  if (atoms.empty()) {
    error("RGPOT needs ATOMS or SPECIES1, SPECIES2, ...");
  }
  {
    std::vector<unsigned> idx;
    idx.reserve(atoms.size());
    for (const AtomNumber &a : atoms) {
      idx.push_back(a.index());
    }
    std::sort(idx.begin(), idx.end());
    if (std::adjacent_find(idx.begin(), idx.end()) != idx.end()) {
      error("an atom appears twice in ATOMS or SPECIESn");
    }
  }

  std::string name;
  parse("POTENTIAL", name);
  std::vector<std::string> words;
  parseVector("OPTIONS", words);
  double cutoff = -1.0;
  parse("CUTOFF", cutoff);
  double u0 = -1.0;
  double psi = -1.0;
  parse("U0", u0);
  parse("PSI", psi);
  std::vector<std::string> elements;
  parseVector("ELEMENTS", elements);
  parseFlag("NOPBC", nopbc_);
  parse("VACUUM", vacuum_);
  std::string virial;
  parse("VIRIAL", virial);
  checkRead();

  if (virial == "FD") {
    virial_fd_ = true;
  } else if (virial != "AUTO") {
    error("VIRIAL is AUTO or FD, not " + virial);
  }

  rgpot::spec::Options options;
  const bool legacy = name.empty();
  if (legacy) {
    if (!words.empty()) {
      error("OPTIONS needs POTENTIAL");
    }
    name = "LJ";
    if (cutoff < 0.0) {
      cutoff = 8.0;
    }
    options.emplace_back("u0", number((u0 < 0.0 ? 1.0 : u0) / energy_per_ev_));
    options.emplace_back("psi", number((psi < 0.0 ? 1.0 : psi) * to_angstrom_));
  } else {
    if (u0 >= 0.0 || psi >= 0.0) {
      error("U0 and PSI belong to the RGPOT form without POTENTIAL; give POTENTIAL=LJ "
            "OPTIONS=u0=...,psi=... instead");
    }
    for (const std::string &w : words) {
      const std::size_t eq = w.find('=');
      if (eq == std::string::npos || eq == 0) {
        error("OPTIONS takes key=value pairs, not " + w);
      }
      options.emplace_back(w.substr(0, eq), w.substr(eq + 1));
    }
  }
  if (cutoff == 0.0) {
    error("CUTOFF must be positive");
  }
  const double cutoff_angstrom = cutoff > 0.0 ? cutoff * to_angstrom_ : 0.0;

  z_.assign(atoms.size(), 1);
  if (!elements.empty()) {
    std::vector<int> zs;
    for (const std::string &el : elements) {
      const int z = rgpot::spec::atomicNumber(el);
      if (z == 0) {
        error(el + " is not an element symbol");
      }
      zs.push_back(z);
    }
    if (zs.size() == 1) {
      z_.assign(atoms.size(), zs[0]);
    } else if (zs.size() == group_size.size() && group_size.size() > 1) {
      std::size_t at = 0;
      for (std::size_t g = 0; g < group_size.size(); ++g) {
        for (std::size_t k = 0; k < group_size[g]; ++k) {
          z_[at++] = zs[g];
        }
      }
    } else if (zs.size() == atoms.size() && group_size.size() == 1) {
      z_ = zs;
    } else {
      error("ELEMENTS needs one symbol, one per SPECIESn group or one per atom of ATOMS");
    }
  } else if (!legacy) {
    error("POTENTIAL needs ELEMENTS");
  }

  if (nopbc_) {
    if (vacuum_ <= 0.0) {
      if (cutoff <= 0.0) {
        error("NOPBC needs VACUUM, the space between periodic images, longer than the "
              "potential's cutoff");
      }
      vacuum_ = cutoff;
    }
    vacuum_ *= to_angstrom_;
  } else if (vacuum_ != 0.0) {
    error("VACUUM needs NOPBC");
  }

  // Every rank checks the line; rank 0 keeps the kernel it evaluates.
  built_ = rgpot::spec::build(name, options, cutoff_angstrom);
  if (!built_.pot) {
    error("RGPOT: " + built_.error);
  }
  stress_ = built_.pot->caps().stress;

  log.printf("  rgpot potential %s with %zu atoms", name.c_str(), atoms.size());
  for (const auto &kv : options) {
    log.printf(" %s=%s", kv.first.c_str(), kv.second.c_str());
  }
  log.printf("\n");
  if (cutoff_angstrom > 0.0) {
    log.printf("  cutoff %f Angstrom when the options name none\n", cutoff_angstrom);
  }
  log.printf("  1 PLUMED length is %.17g Angstrom, 1 eV is %.17g PLUMED energy\n", to_angstrom_,
             energy_per_ev_);
  if (nopbc_) {
    log.printf("  free cluster, %f Angstrom of vacuum between images\n", vacuum_);
  } else {
    log.printf("  periodic box, virial from %s\n",
               (stress_ && !virial_fd_) ? "the stress" : "finite strains");
  }
  if (comm.Get_rank() != 0) {
    built_ = rgpot::spec::Built{};
  }

  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(atoms);
}

int Rgpot::evaluate(const std::vector<double> &xyz, const double *box, double *energy,
                    double *forces, double *stress, int *has_stress, std::string &why) const {
  const std::size_t n = z_.size();
  std::fill(forces, forces + 3 * n, 0.0);
  rgpot::ForceInput in{n, xyz.data(), z_.data(), box};
  rgpot::ForceOut out{forces, 0.0, 0.0, {}, 0};
  try {
    built_.eval(in, &out);
  } catch (const std::exception &e) {
    why = e.what();
    return 1;
  } catch (...) {
    why = "the rgpot potential failed with a non-standard exception";
    return 1;
  }
  *energy = out.energy;
  *has_stress = out.has_stress;
  std::copy(out.stress, out.stress + 9, stress);
  return 0;
}

int Rgpot::strainVirial(const std::vector<double> &xyz, const double *box, double w[9],
                        std::string &why) const {
  // Rows and positions go to r (I + e). The entry e_ij, i >= j, keeps a
  // lower triangular cell lower triangular, and -dE/de_ij is W_ij = W_ji.
  const int rows[6] = {0, 1, 2, 1, 2, 2};
  const int cols[6] = {0, 1, 2, 0, 0, 1};
  const std::size_t n = z_.size();
  std::vector<double> sx(3 * n);
  std::vector<double> sf(3 * n);
  for (int k = 0; k < 6; ++k) {
    const int i = rows[k];
    const int j = cols[k];
    double e_pm[2] = {};
    for (int side = 0; side < 2; ++side) {
      const double e = side == 0 ? kStrain : -kStrain;
      double sbox[9];
      std::copy(box, box + 9, sbox);
      for (int r = 0; r < 3; ++r) {
        sbox[3 * r + j] += box[3 * r + i] * e;
      }
      std::copy(xyz.begin(), xyz.end(), sx.begin());
      for (std::size_t a = 0; a < n; ++a) {
        sx[3 * a + j] += xyz[3 * a + i] * e;
      }
      double stress[9];
      int has_stress = 0;
      if (evaluate(sx, sbox, &e_pm[side], sf.data(), stress, &has_stress, why) != 0) {
        return 1;
      }
    }
    w[3 * i + j] = -(e_pm[0] - e_pm[1]) / (2.0 * kStrain);
    w[3 * j + i] = w[3 * i + j];
  }
  return 0;
}

void Rgpot::calculate() {
  const std::vector<Vector> &positions = getPositions();
  const std::size_t n = positions.size();
  // energy, then the 3n forces, then the box derivatives, in eV and Angstrom.
  std::vector<double> result(1 + 3 * n + 9, 0.0);
  int status = 0;
  std::string why;
  if (comm.Get_rank() == 0) {
    std::vector<double> xyz(3 * n);
    for (std::size_t i = 0; i < n; ++i) {
      for (int d = 0; d < 3; ++d) {
        xyz[3 * i + d] = positions[i][d] * to_angstrom_;
      }
    }
    double box[9] = {};
    if (nopbc_) {
      double lo[3];
      double hi[3];
      for (int d = 0; d < 3; ++d) {
        lo[d] = std::numeric_limits<double>::max();
        hi[d] = std::numeric_limits<double>::lowest();
        for (std::size_t i = 0; i < n; ++i) {
          lo[d] = std::min(lo[d], xyz[3 * i + d]);
          hi[d] = std::max(hi[d], xyz[3 * i + d]);
        }
        box[4 * d] = hi[d] - lo[d] + vacuum_;
        for (std::size_t i = 0; i < n; ++i) {
          xyz[3 * i + d] += 0.5 * vacuum_ - lo[d];
        }
      }
    } else {
      const Tensor &cell = getBox();
      Tensor h;
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          h(r, c) = cell(r, c) * to_angstrom_;
          box[3 * r + c] = h(r, c);
        }
      }
      const double volume = std::fabs(h.determinant());
      if (!(volume > 0.0)) {
        status = 1;
        why = "RGPOT needs a periodic box; give NOPBC and VACUUM for a cluster";
      } else {
        // x = f h with fractional f in [0, 1).
        const Tensor hinv = inverse(h);
        for (std::size_t i = 0; i < n; ++i) {
          Vector x(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
          Vector f = matmul(x, hinv);
          for (int d = 0; d < 3; ++d) {
            f[d] -= std::floor(f[d]);
          }
          x = matmul(f, h);
          for (int d = 0; d < 3; ++d) {
            xyz[3 * i + d] = x[d];
          }
        }
      }
    }
    double stress[9] = {};
    int has_stress = 0;
    if (status == 0) {
      status = evaluate(xyz, box, &result[0], &result[1], stress, &has_stress, why);
    }
    if (status == 0 && !nopbc_) {
      double *w = &result[1 + 3 * n];
      if (has_stress != 0 && !virial_fd_) {
        const double volume = std::fabs(Tensor(box[0], box[1], box[2], box[3], box[4], box[5],
                                               box[6], box[7], box[8])
                                            .determinant());
        for (int k = 0; k < 9; ++k) {
          w[k] = -volume * stress[k];
        }
      } else {
        status = strainVirial(xyz, box, w, why);
      }
    }
  }
  comm.Bcast(status, 0);
  if (status != 0) {
    int len = static_cast<int>(why.size());
    comm.Bcast(len, 0);
    why.resize(static_cast<std::size_t>(len));
    if (len > 0) {
      comm.Bcast(&why[0], len, 0);
    }
    error("RGPOT could not evaluate this geometry: " + why);
  }
  comm.Bcast(result.data(), static_cast<int>(result.size()), 0);

  setValue(result[0] * energy_per_ev_);
  const double force_scale = energy_per_ev_ * to_angstrom_;
  for (std::size_t i = 0; i < n; ++i) {
    const double *f = &result[1 + 3 * i];
    setAtomsDerivatives(static_cast<int>(i),
                        Vector(-f[0] * force_scale, -f[1] * force_scale, -f[2] * force_scale));
  }
  if (nopbc_) {
    setBoxDerivativesNoPbc();
  } else {
    const double *w = &result[1 + 3 * n];
    Tensor virial;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        virial(r, c) = w[3 * r + c] * energy_per_ev_;
      }
    }
    setBoxDerivatives(virial);
  }
}

} // namespace colvar
} // namespace PLMD
