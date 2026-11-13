// MIT License
// Copyright 2023--present rgpot developers
//
// C ABI over OrcaPot. Linked into librgpot and into liborca_engine.

#include "rgpot/OrcaPot/orca_c_abi.h"

#include "rgpot/OrcaPot/OrcaPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

struct RgpotOrcaPot {
  std::unique_ptr<rgpot::OrcaPot> pot;
  std::string error;
};

namespace {

void set_error(RgpotOrcaPot *pot, const char *text) {
  if (!pot) {
    return;
  }
  pot->error = text ? text : "";
}

} // namespace

extern "C" {

int rgpot_orca_abi_version(void) { return RGPOT_ORCA_ABI_VERSION; }

int rgpot_orca_available(void) { return 1; }

RgpotOrcaPot *rgpot_orca_create(const RgpotOrcaConfig *cfg, char *errbuf,
                                size_t errlen) {
  try {
    rgpot::OrcaConfig c;
    if (cfg) {
      c.charge = cfg->charge;
      if (cfg->multiplicity > 0) {
        c.multiplicity = cfg->multiplicity;
      }
      if (cfg->method && cfg->method[0] != '\0') {
        c.method = cfg->method;
      }
      if (cfg->basis && cfg->basis[0] != '\0') {
        c.basis = cfg->basis;
      }
      if (cfg->keywords) {
        c.keywords = cfg->keywords;
      }
      if (cfg->executable) {
        c.executable = cfg->executable;
      }
      c.reuse_gbw = cfg->reuse_gbw != 0;
      c.nprocs = cfg->nprocs;
      c.maxcore_mb = cfg->maxcore_mb;
    }
    auto *out = new RgpotOrcaPot;
    out->pot = std::make_unique<rgpot::OrcaPot>(c);
    return out;
  } catch (const std::exception &ex) {
    if (errbuf && errlen) {
      std::snprintf(errbuf, errlen, "%s", ex.what());
    }
    return nullptr;
  } catch (...) {
    if (errbuf && errlen) {
      std::snprintf(errbuf, errlen, "unknown error creating ORCA pot");
    }
    return nullptr;
  }
}

void rgpot_orca_destroy(RgpotOrcaPot *pot) { delete pot; }

void rgpot_orca_reset(RgpotOrcaPot *pot) {
  if (pot && pot->pot) {
    pot->pot->reset();
  }
}

int rgpot_orca_reused_wavefunction(const RgpotOrcaPot *pot) {
  if (!pot || !pot->pot) {
    return 0;
  }
  return pot->pot->reusedWavefunction() ? 1 : 0;
}

const char *rgpot_orca_last_error(const RgpotOrcaPot *pot) {
  if (!pot) {
    return "null ORCA potential";
  }
  return pot->error.c_str();
}

int rgpot_orca_calculate(RgpotOrcaPot *pot, long nAtoms,
                         const double *positions, const int *atomicNrs,
                         double *forces, double *energy, double *variance,
                         const double *box) {
  if (!pot || !pot->pot || !positions || !atomicNrs || !forces || !energy ||
      !box || nAtoms <= 0) {
    set_error(pot, "rgpot_orca_calculate: invalid argument");
    return 1;
  }
  try {
    const int n = static_cast<int>(nAtoms);
    rgpot::types::AtomMatrix pos(n, 3);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j) {
        pos(i, j) = positions[static_cast<std::size_t>(i * 3 + j)];
      }
    }
    std::vector<int> atm(atomicNrs, atomicNrs + n);
    std::array<std::array<double, 3>, 3> cell{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cell[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
            box[i * 3 + j];
      }
    }
    auto [e, f, var] = (*pot->pot)(pos, atm, cell);
    *energy = e;
    if (variance) {
      *variance = var;
    }
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < 3; ++j) {
        forces[static_cast<std::size_t>(i * 3 + j)] = f(i, j);
      }
    }
    set_error(pot, "");
    return 0;
  } catch (const std::exception &ex) {
    set_error(pot, ex.what());
    return 2;
  } catch (...) {
    set_error(pot, "unknown exception in rgpot_orca_calculate");
    return 2;
  }
}

} // extern "C"
