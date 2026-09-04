#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <string>
#include <vector>

#include <dftd3.h>

#include "rgpot/Potential.hpp"

namespace rgpot {

enum class D3Damping { BJ, Zero };

struct D3Config {
  D3Damping damping = D3Damping::BJ;
  std::string functional = "pbe";
  /// Axilrod-Teller-Muto E(3). Literature D3-ATM default is on; the flag
  /// is an explicit constructor argument so tests can pin both values.
  bool atm = true;
};

/**
 * Linked DFT-D3 pot (NEEDED libs-dftd3).
 *
 * s-dftd3 exposes a stable **ISO_C_BINDING** C API (``dftd3.h`` /
 * ``s-dftd3.h``): opaque ``dftd3_error`` / ``dftd3_structure`` /
 * ``dftd3_model`` /
 * ``dftd3_param`` handles with explicit create/update/destroy. We test and
 * use that C contract. ``atm`` maps to the library's ``s9`` switch on
 * ``dftd3_load_{rational,zero}_damping``. Each ``D3Pot`` owns its handles;
 * first force builds the structure and model, later forces call
 * ``dftd3_update_structure``. Do not share one instance across threads.
 */
class D3Pot : public Potential<D3Pot> {
public:
  D3Pot();
  explicit D3Pot(const D3Config &config);
  ~D3Pot();

  D3Pot(const D3Pot &) = delete;
  D3Pot &operator=(const D3Pot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

  /// Native handles are per-instance; concurrent use needs one instance
  /// per thread.
  [[nodiscard]] PotCaps caps() const noexcept override {
    return {.reentrancy = Reentrancy::PerInstance};
  }

  [[nodiscard]] const D3Config &config() const noexcept { return m_config; }

private:
  D3Config m_config;

  mutable dftd3_error m_error = nullptr;
  mutable dftd3_structure m_mol = nullptr;
  mutable dftd3_model m_model = nullptr;
  mutable dftd3_param m_param = nullptr;
  mutable bool m_initialized = false;
  mutable std::vector<double> m_pos_bohr;

  void initHandles();
  void loadParam();
  void checkError(const char *what) const;
};

} // namespace rgpot
