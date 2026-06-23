#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <vector>

#include <xtb.h>

#include "rgpot/Potential.hpp"

namespace rgpot {

enum class GFNMethod { GFNFF, GFN0xTB, GFN1xTB, GFN2xTB };

struct XTBConfig {
  GFNMethod method = GFNMethod::GFN2xTB;
  double accuracy = 1.0;
  double electronic_temperature = 300.0; // Kelvin
  int max_iterations = 250;
  double charge = 0.0;
  int uhf = 0;
};

class XTBPot : public Potential<XTBPot> {
public:
  XTBPot();
  explicit XTBPot(const XTBConfig &config);
  ~XTBPot();

  XTBPot(const XTBPot &) = delete;
  XTBPot &operator=(const XTBPot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  XTBConfig m_config;

  mutable xtb_TEnvironment m_env = nullptr;
  mutable xtb_TCalculator m_calc = nullptr;
  mutable xtb_TMolecule m_mol = nullptr;
  mutable xtb_TResults m_res = nullptr;
  mutable bool m_initialized = false;
  mutable std::vector<double> m_pos_bohr; //!< Preallocated position buffer.

  void initHandles();
  void loadParametrisation() const;
};

} // namespace rgpot
