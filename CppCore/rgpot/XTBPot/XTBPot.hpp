#pragma once
// MIT License
// Copyright 2023--present rgpot developers

#include <vector>

#include <xtb.h>

#include "rgpot/Potential.hpp"
#include "rgpot/XTBPot/XTBConfig.hpp"

namespace rgpot {

/**
 * Linked GFN-xTB pot (NEEDED libxtb).
 *
 * libxtb exposes a stable **ISO_C_BINDING** C API (``xtb.h``): opaque
 * ``xtb_TEnvironment`` / ``xtb_TCalculator`` / ``xtb_TMolecule`` /
 * ``xtb_TResults`` handles with explicit create/update/destroy. We test and
 * use that C contract—not free-form Fortran ABI. Each ``XTBPot`` owns its
 * handles; first force builds the molecule, later forces call
 * ``xtb_updateMolecule``. ``xtb_releaseOutput`` is called on create (library
 * recommendation for multi-env processes). Do not share one instance across
 * threads; distinct instances may be used from different threads if the
 * loaded libxtb build is reentrant for independent handle sets.
 *
 * For hosts that must not NEEDED libxtb, use ``XTBDlopen`` +
 * ``libxtb_engine.so`` instead (always built; no ``-Dwith_xtb`` required).
 */
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
