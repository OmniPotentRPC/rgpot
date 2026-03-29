// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/XTBPot/XTBPot.hpp"
#include "rgpot/simd_ops.hpp"
#include "rgpot/units.hpp"

#include <stdexcept>
#include <string>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

XTBPot::XTBPot() : XTBPot(XTBConfig{}) {}

XTBPot::XTBPot(const XTBConfig &config)
    : Potential(PotType::XTB), m_config(config) {
  initHandles();
}

void XTBPot::initHandles() {
  m_env = xtb_newEnvironment();
  if (!m_env) {
    throw std::runtime_error("Failed to create xtb environment");
  }
  xtb_setVerbosity(m_env, XTB_VERBOSITY_MUTED);
  xtb_releaseOutput(m_env);

  m_calc = xtb_newCalculator();
  if (!m_calc) {
    xtb_delEnvironment(&m_env);
    throw std::runtime_error("Failed to create xtb calculator");
  }

  m_res = xtb_newResults();
  if (!m_res) {
    xtb_delCalculator(&m_calc);
    xtb_delEnvironment(&m_env);
    throw std::runtime_error("Failed to create xtb results");
  }
}

XTBPot::~XTBPot() {
  if (m_res)
    xtb_delResults(&m_res);
  if (m_calc)
    xtb_delCalculator(&m_calc);
  if (m_mol)
    xtb_delMolecule(&m_mol);
  if (m_env) {
    xtb_releaseOutput(m_env);
    xtb_delEnvironment(&m_env);
  }
}

void XTBPot::loadParametrisation() const {
  switch (m_config.method) {
  case GFNMethod::GFNFF:
    xtb_loadGFNFF(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN0xTB:
    xtb_loadGFN0xTB(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN1xTB:
    xtb_loadGFN1xTB(m_env, m_mol, m_calc, nullptr);
    break;
  case GFNMethod::GFN2xTB:
    xtb_loadGFN2xTB(m_env, m_mol, m_calc, nullptr);
    break;
  }
  xtb_setAccuracy(m_env, m_calc, m_config.accuracy);
  xtb_setElectronicTemp(m_env, m_calc, m_config.electronic_temperature);
  xtb_setMaxIter(m_env, m_calc, m_config.max_iterations);
}

void XTBPot::forceImpl(const ForceInput &in, ForceOut *out) const {
  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;
  const bool periodicity[3] = {false, false, false};

  // Reuse preallocated buffer, resize only when atom count changes
  m_pos_bohr.resize(n3);
  simd::scale(m_pos_bohr.data(), in.pos, ANGSTROM_TO_BOHR, n3);

  double box_bohr[9];
  simd::scale(box_bohr, in.box, ANGSTROM_TO_BOHR, 9);

  if (!m_initialized) {
    double charge = m_config.charge;
    int uhf = m_config.uhf;
    m_mol = xtb_newMolecule(m_env, &intN, in.atmnrs, m_pos_bohr.data(),
                            &charge, &uhf, box_bohr, periodicity);
    loadParametrisation();
    m_initialized = true;
  } else {
    xtb_updateMolecule(m_env, m_mol, m_pos_bohr.data(), box_bohr);
  }

  xtb_singlepoint(m_env, m_mol, m_calc, m_res);

  if (xtb_checkEnvironment(m_env) != 0) {
    char err_msg[512];
    xtb_getError(m_env, err_msg, nullptr);
    throw std::runtime_error(std::string("xTB Error: ") + err_msg);
  }

  double energy_hartree;
  xtb_getEnergy(m_env, m_res, &energy_hartree);
  out->energy = energy_hartree * HARTREE_TO_EV;

  // Write gradient directly into output buffer, then convert in-place
  xtb_getGradient(m_env, m_res, out->F);
  simd::scale_inplace(out->F, NEG_GRAD_TO_FORCE, n3);

  out->variance = 0.0;
}

} // namespace rgpot
