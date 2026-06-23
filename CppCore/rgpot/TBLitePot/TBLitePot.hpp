#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <vector>

#include <tblite.h>

#include "rgpot/Potential.hpp"

namespace rgpot {

enum class TBLiteMethod { GFN1, GFN2, IPEA1 };

struct TBLiteConfig {
  TBLiteMethod method = TBLiteMethod::GFN2;
  double accuracy = 1.0;
  double electronic_temperature = 300.0; // Kelvin
  int max_iterations = 250;
  double charge = 0.0;
  int uhf = 0;
};

class TBLitePot : public Potential<TBLitePot> {
public:
  TBLitePot();
  explicit TBLitePot(const TBLiteConfig &config);
  ~TBLitePot();

  TBLitePot(const TBLitePot &) = delete;
  TBLitePot &operator=(const TBLitePot &) = delete;

  void forceImpl(const ForceInput &in, ForceOut *out) const override;

private:
  TBLiteConfig m_config;

  mutable tblite_context m_ctx = nullptr;
  mutable tblite_calculator m_calc = nullptr;
  mutable tblite_structure m_mol = nullptr;
  mutable tblite_result m_res = nullptr;
  mutable bool m_initialized = false;
  mutable std::vector<double> m_pos_bohr; //!< Preallocated position buffer.

  void initHandles();
  void createCalculator() const;
};

} // namespace rgpot
