// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/units.hpp"

#include <stdexcept>
#include <string>

namespace rgpot {

using units::ANGSTROM_TO_BOHR;
using units::HARTREE_TO_EV;
using units::KB_HARTREE;
using units::NEG_GRAD_TO_FORCE;

TBLitePot::TBLitePot() : TBLitePot(TBLiteConfig{}) {}

TBLitePot::TBLitePot(const TBLiteConfig &config)
    : Potential(PotType::TBLite), m_config(config) {
  initHandles();
}

void TBLitePot::initHandles() {
  m_ctx = tblite_new_context();
  if (!m_ctx) {
    throw std::runtime_error("Failed to create tblite context");
  }
  tblite_set_context_verbosity(m_ctx, 0);

  m_res = tblite_new_result();
  if (!m_res) {
    tblite_delete_context(&m_ctx);
    throw std::runtime_error("Failed to create tblite result");
  }
}

TBLitePot::~TBLitePot() {
  if (m_res)
    tblite_delete_result(&m_res);
  if (m_calc)
    tblite_delete_calculator(&m_calc);
  if (m_mol)
    tblite_delete_structure(&m_mol);
  if (m_ctx)
    tblite_delete_context(&m_ctx);
}

void TBLitePot::createCalculator() const {
  switch (m_config.method) {
  case TBLiteMethod::GFN1:
    m_calc = tblite_new_gfn1_calculator(m_ctx, m_mol);
    break;
  case TBLiteMethod::GFN2:
    m_calc = tblite_new_gfn2_calculator(m_ctx, m_mol);
    break;
  case TBLiteMethod::IPEA1:
    m_calc = tblite_new_ipea1_calculator(m_ctx, m_mol);
    break;
  }

  if (tblite_check_context(m_ctx) != 0) {
    char err_msg[512];
    tblite_get_context_error(m_ctx, err_msg, nullptr);
    throw std::runtime_error(std::string("tblite calculator error: ") +
                             err_msg);
  }

  tblite_set_calculator_accuracy(m_ctx, m_calc, m_config.accuracy);
  tblite_set_calculator_max_iter(m_ctx, m_calc, m_config.max_iterations);
  double etemp_hartree = m_config.electronic_temperature * KB_HARTREE;
  tblite_set_calculator_temperature(m_ctx, m_calc, etemp_hartree);
}

void TBLitePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  int intN = static_cast<int>(in.nAtoms);
  const size_t n3 = 3 * in.nAtoms;

  // Reuse preallocated buffer
  m_pos_bohr.resize(n3);
  for (size_t i = 0; i < n3; ++i) {
    m_pos_bohr[i] = in.pos[i] * ANGSTROM_TO_BOHR;
  }

  double box_bohr[9];
  for (int i = 0; i < 9; ++i) {
    box_bohr[i] = in.box[i] * ANGSTROM_TO_BOHR;
  }

  if (!m_initialized) {
    tblite_error err = tblite_new_error();
    double charge = m_config.charge;
    int uhf = m_config.uhf;
    m_mol = tblite_new_structure(err, intN, in.atmnrs, m_pos_bohr.data(),
                                 &charge, &uhf, box_bohr, nullptr);
    if (tblite_check_error(err) != 0) {
      char err_msg[512];
      tblite_get_error(err, err_msg, nullptr);
      tblite_delete_error(&err);
      throw std::runtime_error(std::string("tblite structure error: ") +
                               err_msg);
    }
    tblite_delete_error(&err);
    createCalculator();
    m_initialized = true;
  } else {
    tblite_error err = tblite_new_error();
    tblite_update_structure_geometry(err, m_mol, m_pos_bohr.data(), box_bohr);
    if (tblite_check_error(err) != 0) {
      char err_msg[512];
      tblite_get_error(err, err_msg, nullptr);
      tblite_delete_error(&err);
      throw std::runtime_error(std::string("tblite update error: ") + err_msg);
    }
    tblite_delete_error(&err);
  }

  tblite_get_singlepoint(m_ctx, m_mol, m_calc, m_res);

  if (tblite_check_context(m_ctx) != 0) {
    char err_msg[512];
    tblite_get_context_error(m_ctx, err_msg, nullptr);
    throw std::runtime_error(std::string("tblite SCF error: ") + err_msg);
  }

  tblite_error err = tblite_new_error();

  double energy_hartree;
  tblite_get_result_energy(err, m_res, &energy_hartree);
  out->energy = energy_hartree * HARTREE_TO_EV;

  // Write gradient directly into output, convert in-place
  tblite_get_result_gradient(err, m_res, out->F);
  for (size_t i = 0; i < n3; ++i) {
    out->F[i] *= NEG_GRAD_TO_FORCE;
  }

  out->variance = 0.0;
  tblite_delete_error(&err);
}

} // namespace rgpot
