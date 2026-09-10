#pragma once
// MIT License — Skala XC on the NWChem C ABI engine

#include <string>

namespace rgpot {

/**
 * Kohn-Sham DFT through libnwchemc, with Skala as the XC keyword.
 *
 * Skala is not a standalone PES. Energy and forces come from
 * nwchemc_energy_gradient after NWChemParams.dft.xc is set.
 */
struct SkalaConfig {
  std::string xc = "skala-1.1";
  std::string basis = "def2-tzvp";
  int charge = 0;
  /// Spin multiplicity 2S+1 (NWChemParams.multiplicity).
  int multiplicity = 1;
  std::string engine_path;
  std::string nwchem_root;
};

} // namespace rgpot
