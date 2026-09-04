// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/D3Pot/D3Pot.hpp"
#include "rgpot/units.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::NEG_GRAD_TO_FORCE;

D3Pot::D3Pot() : D3Pot(D3Config{}) {}

D3Pot::D3Pot(const D3Config &config)
    : Potential(PotType::D3), m_config(config) {
  if (m_config.functional.empty()) {
    throw std::invalid_argument("D3Pot functional key must be non-empty");
  }
  initHandles();
}

void D3Pot::initHandles() {
  m_error = dftd3_new_error();
  if (!m_error) {
    throw std::runtime_error("Failed to create dftd3 error handle");
  }
  loadParam();
}

void D3Pot::loadParam() {
  std::vector<char> method(m_config.functional.begin(),
                           m_config.functional.end());
  method.push_back('\0');
  switch (m_config.damping) {
  case D3Damping::BJ:
    m_param = dftd3_load_rational_damping(m_error, method.data(), m_config.atm);
    break;
  case D3Damping::Zero:
    m_param = dftd3_load_zero_damping(m_error, method.data(), m_config.atm);
    break;
  }
  checkError("dftd3 load damping");
  if (!m_param) {
    throw std::runtime_error("Failed to load dftd3 damping parameters for '" +
                             m_config.functional + "'");
  }
}

void D3Pot::checkError(const char *what) const {
  if (dftd3_check_error(m_error) == 0) {
    return;
  }
  char err_msg[512] = {};
  dftd3_get_error(m_error, err_msg, nullptr);
  throw std::runtime_error(std::string(what) + ": " + err_msg);
}

D3Pot::~D3Pot() {
  if (m_param)
    dftd3_delete_param(&m_param);
  if (m_model)
    dftd3_delete_model(&m_model);
  if (m_mol)
    dftd3_delete_structure(&m_mol);
  if (m_error)
    dftd3_delete_error(&m_error);
}

void D3Pot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (in.nAtoms == 0) {
    throw std::invalid_argument("D3Pot requires at least one atom");
  }
  const int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;

  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  if (!m_initialized) {
    m_mol = dftd3_new_structure(m_error, intN, in.atmnrs, m_pos_bohr.data(),
                                nullptr, nullptr);
    checkError("dftd3 new structure");
    m_model = dftd3_new_d3_model(m_error, m_mol);
    checkError("dftd3 new D3 model");
    m_initialized = true;
  } else {
    dftd3_update_structure(m_error, m_mol, m_pos_bohr.data(), nullptr);
    checkError("dftd3 update structure");
  }

  double energy_hartree = 0.0;
  dftd3_get_dispersion(m_error, m_mol, m_model, m_param, &energy_hartree,
                       out->F, nullptr);
  checkError("dftd3 get dispersion");

  out->energy = energy_hartree * HARTREE_TO_EV;
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }
  out->variance = 0.0;
}

} // namespace rgpot
