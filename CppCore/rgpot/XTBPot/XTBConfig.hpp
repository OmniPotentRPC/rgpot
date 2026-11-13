#pragma once
// MIT License
// Copyright 2023--present rgpot developers
//
// Config POD shared by linked XTBPot and the XTBDlopen frontend.
// No libxtb headers: hosts may compile XTBDlopen without -Dwith_xtb.

#include <string>

namespace rgpot {

enum class GFNMethod { GFNFF, GFN0xTB, GFN1xTB, GFN2xTB };

struct XTBConfig {
  GFNMethod method = GFNMethod::GFN2xTB;
  double accuracy = 1.0;
  double electronic_temperature = 300.0; // Kelvin
  int max_iterations = 250;
  double charge = 0.0;
  /** Number of unpaired electrons (libxtb uhf; multiplicity 2 => uhf 1). */
  int uhf = 0;
  /**
   * Largest atom move, in Angstrom, that still reuses the previous SCC
   * wavefunction as the next guess. A larger move drops the results
   * object, and the next single point uses xtb's own guess. Zero or a
   * negative value disables reuse.
   */
  double warm_move = 1.0;
  /**
   * Boundary conditions. 0 is a cluster. A positive value is periodic on
   * every axis of a non-singular cell. Negative (the default) is periodic
   * only when the atoms span more than half of every axis. libxtb's own
   * GFN1-xTB and GFN2-xTB calculators do not support that cell: GFN2 stops
   * with "Multipoles not available with PBC", and the command-line driver
   * then aborts. Periodic GFN1 and GFN2 are `TBLitePot`.
   */
  int periodic = -1;
};

/** Optional engine path for the dlopen frontend (also RGPOT_XTB_ENGINE). */
struct XTBDlopenConfig {
  XTBConfig xtb{};
  std::string engine_path;
};

} // namespace rgpot
