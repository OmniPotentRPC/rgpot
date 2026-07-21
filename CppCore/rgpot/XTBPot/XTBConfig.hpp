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
};

/** Optional engine path for the dlopen frontend (also RGPOT_XTB_ENGINE). */
struct XTBDlopenConfig {
  XTBConfig xtb{};
  std::string engine_path;
};

} // namespace rgpot
