// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/TBLitePot/TBLitePot.hpp"
#include "rgpot/stress.hpp"
#include "rgpot/units.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

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

namespace {

// tblite 0.6 adds an optional xtb config. Null keeps the 0.5 defaults.
tblite_calculator newGfnCalculator(tblite_context ctx, tblite_structure mol,
                                   TBLiteMethod method) {
#ifdef RGPOT_TBLITE_XTB_CONFIG
  auto call = [](auto make, tblite_context c, tblite_structure m) {
    return make(c, m, nullptr);
  };
#else
  auto call = [](auto make, tblite_context c, tblite_structure m) {
    return make(c, m);
  };
#endif
  switch (method) {
  case TBLiteMethod::GFN1:
    return call(tblite_new_gfn1_calculator, ctx, mol);
  case TBLiteMethod::GFN2:
    return call(tblite_new_gfn2_calculator, ctx, mol);
  case TBLiteMethod::IPEA1:
    return call(tblite_new_ipea1_calculator, ctx, mol);
  }
  throw std::logic_error("unknown tblite method");
}

} // namespace

void TBLitePot::createCalculator() const {
  m_calc = newGfnCalculator(m_ctx, m_mol, m_config.method);

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

// Only GFN1 reports stress. Its energy is smooth under strain, and the
// central difference converges to the same value for steps from 1e-3 to
// 8e-3. The GFN2 energy of tblite 0.4 is not: a hard dispersion cutoff makes
// the central difference drift linearly with the step, by 0.2 percent per
// 1e-3, so no stress of that model can be stated to better than that. GFN2
// and IPEA1 refuse the request instead of publishing a number that is only
// roughly the derivative of the energy.

// dE/d(strain) from central differences of the energy, in eV, in the Voigt
// order xx, yy, zz, yz, xz, xy.
//
// tblite 0.4 returns a virial, but for a periodic cell it disagrees with the
// derivative of its own energy: the H0 self-image pairs carry the sigma
// factor of an ordinary pair and the multipole Ewald sum has a wrong
// reciprocal-space term, both fixed upstream in later releases. A silicon
// cell of eight atoms differs from the energy derivative by 3 percent for
// GFN1 and by up to 20 percent for GFN2. The derivative of the energy is the
// quantity the stress is defined from, so it is taken from the energy, which
// costs twelve single points and only when stress is requested.
void TBLitePot::strainDerivative(const std::vector<double> &pos_bohr,
                                 const double box_bohr[9], const int *numbers,
                                 double strain[6]) const {
  constexpr double step = 5e-4;
  constexpr int rows[6] = {0, 1, 2, 1, 0, 0};
  constexpr int cols[6] = {0, 1, 2, 2, 2, 1};
  const std::size_t n = pos_bohr.size() / 3;
  std::vector<double> moved(pos_bohr.size());
  double cell[9];

  auto energyAt = [&](const double transform[9]) {
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < 3; ++c) {
        moved[3 * i + c] = pos_bohr[3 * i] * transform[c] +
                           pos_bohr[3 * i + 1] * transform[3 + c] +
                           pos_bohr[3 * i + 2] * transform[6 + c];
      }
    }
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        cell[3 * r + c] = box_bohr[3 * r] * transform[c] +
                          box_bohr[3 * r + 1] * transform[3 + c] +
                          box_bohr[3 * r + 2] * transform[6 + c];
      }
    }
    // A calculator is set up for one lattice, so each strained cell gets
    // its own structure and calculator, as a new potential would.
    tblite_error err = tblite_new_error();
    double charge = m_config.charge;
    int uhf = m_config.uhf;
    tblite_structure mol =
        tblite_new_structure(err, static_cast<int>(n), numbers, moved.data(),
                             &charge, &uhf, cell, nullptr);
    const bool built = tblite_check_error(err) == 0;
    tblite_delete_error(&err);
    if (!built) {
      throw std::runtime_error("tblite strained structure failed");
    }
    tblite_calculator calc = newGfnCalculator(m_ctx, mol, m_config.method);
    tblite_set_calculator_accuracy(m_ctx, calc, m_config.accuracy);
    tblite_set_calculator_max_iter(m_ctx, calc, m_config.max_iterations);
    tblite_set_calculator_temperature(
        m_ctx, calc, m_config.electronic_temperature * KB_HARTREE);
    tblite_result result = tblite_new_result();
    tblite_get_singlepoint(m_ctx, mol, calc, result);
    const bool failed = tblite_check_context(m_ctx) != 0;
    std::string message;
    if (failed) {
      char text[512];
      tblite_get_context_error(m_ctx, text, nullptr);
      message = text;
    }
    double energy = 0.0;
    if (!failed) {
      tblite_error read = tblite_new_error();
      tblite_get_result_energy(read, result, &energy);
      tblite_delete_error(&read);
    }
    tblite_delete_result(&result);
    tblite_delete_calculator(&calc);
    tblite_delete_structure(&mol);
    if (failed) {
      throw std::runtime_error("tblite stress SCF error: " + message);
    }
    return energy;
  };

  for (int k = 0; k < 6; ++k) {
    double energies[2] = {0.0, 0.0};
    for (int side = 0; side < 2; ++side) {
      const double delta = side == 0 ? -step : step;
      double transform[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      const int a = rows[k];
      const int b = cols[k];
      if (a == b) {
        transform[3 * a + b] += delta;
      } else {
        transform[3 * a + b] += 0.5 * delta;
        transform[3 * b + a] += 0.5 * delta;
      }
      energies[side] = energyAt(transform);
    }
    strain[k] = (energies[1] - energies[0]) / (2.0 * step) * HARTREE_TO_EV;
  }
}

void TBLitePot::forceImpl(const ForceInput &in, ForceOut *out) const {
  if (in.compute_stress && !caps().stress) {
    throw std::invalid_argument(
        "TBLitePot computes stress for GFN1 only: the GFN2 and IPEA1 "
        "energies of tblite are not smooth under strain");
  }
  StressAccumulator stress(in, out);
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
    // The result keeps the wavefunction of the failed SCF, and tblite
    // would start the next one from it.
    tblite_delete_result(&m_res);
    m_res = tblite_new_result();
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

  if (double *strain = stress.data()) {
    strainDerivative(m_pos_bohr, box_bohr, in.atmnrs, strain);
    stress.publish(in.box);
  }

  out->variance = 0.0;
  tblite_delete_error(&err);
}

} // namespace rgpot
