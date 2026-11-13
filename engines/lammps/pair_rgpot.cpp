// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright 2023--present rgpot developers
//
// pair_style rgpot. This file includes the LAMMPS headers, so it is
// distributed under the LAMMPS license. The rest of rgpot stays MIT.
//
//   pair_style rgpot cutoff
//   pair_coeff * * u0 psi
//   pair_coeff * * engine /path/to/libengine.so
//
// cutoff, u0 and psi are in the LAMMPS metal units (Angstrom, eV).
// The style evaluates the whole box in one call. It requires a serial
// run: every atom is local. engine loads an rgpot_engine_* plugin
// (the in-tree LJ engine, or a metatomic / UMA engine).

#include "pair.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "lammpsplugin.h"
#include "memory.h"
#include "neighbor.h"
#include "domain.h"
#include "utils.h"
#include "version.h"

#include "lj_eval.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

class PairRgpot : public Pair {
 public:
  PairRgpot(LAMMPS *lmp);
  ~PairRgpot() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  double cut_global = 0.0;
  double u0 = 1.0;
  double psi = 1.0;
  bool use_engine = false;
  char engine_path[512] = {};
};

PairRgpot::PairRgpot(LAMMPS *lmp) : Pair(lmp) {}

PairRgpot::~PairRgpot() {
  if (copymode) {
    return;
  }
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairRgpot::settings(int narg, char **arg) {
  if (narg != 1) {
    error->all(FLERR, "pair_style rgpot takes the cutoff");
  }
  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (!(cut_global > 0.0)) {
    error->all(FLERR, "pair_style rgpot cutoff must be positive");
  }
}

void PairRgpot::coeff(int narg, char **arg) {
  if (narg != 4) {
    error->all(FLERR, "pair_coeff for rgpot is * * u0 psi, or * * engine path");
  }
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0) {
    error->all(FLERR, "pair_style rgpot only supports pair_coeff * *");
  }
  if (!allocated) {
    allocated = 1;
    memory->create(setflag, atom->ntypes + 1, atom->ntypes + 1, "pair:setflag");
    memory->create(cutsq, atom->ntypes + 1, atom->ntypes + 1, "pair:cutsq");
    for (int i = 1; i <= atom->ntypes; ++i) {
      for (int j = i; j <= atom->ntypes; ++j) {
        setflag[i][j] = 0;
      }
    }
  }
  if (strcmp(arg[2], "engine") == 0) {
    use_engine = true;
    std::snprintf(engine_path, sizeof(engine_path), "%s", arg[3]);
  } else {
    use_engine = false;
    u0 = utils::numeric(FLERR, arg[2], false, lmp);
    psi = utils::numeric(FLERR, arg[3], false, lmp);
    if (!(psi > 0.0)) {
      error->all(FLERR, "pair_style rgpot psi must be positive");
    }
  }
  for (int i = 1; i <= atom->ntypes; ++i) {
    for (int j = i; j <= atom->ntypes; ++j) {
      setflag[i][j] = 1;
    }
  }
}

void PairRgpot::init_style() { neighbor->add_request(this); }

double PairRgpot::init_one(int, int) { return cut_global; }

void PairRgpot::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (atom->nlocal != static_cast<int>(atom->natoms)) {
    error->all(FLERR, "pair_style rgpot evaluates the whole box and needs a serial run");
  }
  const int n = atom->nlocal;
  std::vector<double> xyz(static_cast<size_t>(n) * 3);
  std::vector<int> types(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    xyz[static_cast<size_t>(3 * i)] = atom->x[i][0];
    xyz[static_cast<size_t>(3 * i + 1)] = atom->x[i][1];
    xyz[static_cast<size_t>(3 * i + 2)] = atom->x[i][2];
    types[static_cast<size_t>(i)] = atom->type[i];
  }
  const double box[9] = {domain->xprd, 0.0,         0.0,        domain->xy, domain->yprd,
                         0.0,          domain->xz, domain->yz, domain->zprd};
  std::vector<double> forces(static_cast<size_t>(n) * 3);
  double energy = 0.0;
  if (use_engine) {
    char why[256] = {};
    if (rgpot_engine_eval(engine_path, nullptr, 0, xyz.data(), types.data(),
                          static_cast<size_t>(n), box, forces.data(), &energy, why,
                          sizeof(why)) != 0) {
      error->all(FLERR, why[0] != '\0' ? why : "rgpot engine evaluation failed");
    }
  } else {
    int has_stress = 0;
    double stress[9];
    if (rgpot_lj_eval(xyz.data(), types.data(), static_cast<size_t>(n), box, u0, cut_global, psi,
                      forces.data(), &energy, &has_stress, stress) != 0) {
      error->all(FLERR, "rgpot Lennard-Jones evaluation failed");
    }
  }
  for (int i = 0; i < n; ++i) {
    atom->f[i][0] += forces[static_cast<size_t>(3 * i)];
    atom->f[i][1] += forces[static_cast<size_t>(3 * i + 1)];
    atom->f[i][2] += forces[static_cast<size_t>(3 * i + 2)];
  }
  if (eflag_global) {
    eng_vdwl += energy;
  }
}

static Pair *rgpot_creator(LAMMPS *lmp) { return new PairRgpot(lmp); }

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
  auto *register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "rgpot";
  plugin.info = "rgpot Lennard-Jones or an rgpot_engine plugin";
  plugin.author = "Rohit Goswami (rgoswami@ieee.org)";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&rgpot_creator);
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
