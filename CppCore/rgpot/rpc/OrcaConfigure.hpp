#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file OrcaConfigure.hpp
 * @brief ORCAParams and CommonMethodSpec to ORCAConfig, and the potserv
 * configure() step for ORCAPot.
 *
 * The CommonMethodSpec overlay is lowered here, in rgpot, into ORCA
 * simple-input keywords and % blocks; liborcac only ever sees ORCAConfig.
 * Lowering runs first and the orca arm second, so any field the arm sets
 * wins. Keywords and blocks the overlay produces come before the arm's
 * own, so ORCA reads the arm's last.
 */

#include <algorithm>
#include <cctype>
#include <charconv>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"
#include "rgpot/units.hpp"

namespace rgpot::rpc {

namespace orca_detail {

inline std::string upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return s;
}

inline std::string number(double v) {
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), v);
  return std::string(buf, res.ptr);
}

inline void append(std::string &into, const std::string &text,
                   const char *sep) {
  if (text.empty()) {
    return;
  }
  into += (into.empty() ? "" : sep) + text;
}

inline bool is_libxc(const std::string &name) {
  for (const char *prefix : {"LDA_", "GGA_", "MGGA_", "HYB_"}) {
    if (name.rfind(prefix, 0) == 0) {
      return true;
    }
  }
  return false;
}

/// ORCA's keyword for a libxc functional list, or empty when none matches.
inline std::string libxc_to_orca(std::vector<std::string> names) {
  std::sort(names.begin(), names.end());
  struct Entry {
    std::vector<std::string> libxc;
    const char *orca;
  };
  static const Entry table[] = {
      {{"GGA_C_PBE", "GGA_X_PBE"}, "PBE"},
      {{"GGA_C_LYP", "GGA_X_B88"}, "BLYP"},
      {{"HYB_GGA_XC_B3LYP"}, "B3LYP"},
      {{"HYB_GGA_XC_PBEH"}, "PBE0"},
      {{"MGGA_C_TPSS", "MGGA_X_TPSS"}, "TPSS"},
      {{"MGGA_C_R2SCAN", "MGGA_X_R2SCAN"}, "r2SCAN"},
      {{"HYB_GGA_XC_WB97X"}, "wB97X"},
  };
  for (const auto &e : table) {
    if (e.libxc == names) {
      return e.orca;
    }
  }
  return {};
}

} // namespace orca_detail

/// Lowers the normalized overlay into cfg. Throws std::invalid_argument for
/// a setting ORCA cannot take: plane waves, a k mesh, smearing other than
/// Fermi, or a functional, dispersion or relativity name with no ORCA
/// keyword.
inline void lowerCommonMethod(const ::CommonMethodSpec::Reader &common,
                              ORCAConfig &cfg) {
  using namespace orca_detail;
  auto refuse = [](const std::string &why) {
    throw std::invalid_argument("CommonMethodSpec cannot lower to ORCA: " +
                                why);
  };

  std::vector<std::string> xc;
  for (auto name : common.getXcFunctionals()) {
    xc.emplace_back(name.cStr());
  }
  if (xc.size() == 1 && !is_libxc(xc[0])) {
    cfg.method = xc[0];
  } else if (!xc.empty()) {
    const std::string method = libxc_to_orca(xc);
    if (method.empty()) {
      std::string list;
      for (const auto &n : xc) {
        append(list, n, ", ");
      }
      refuse("no ORCA keyword for the functionals " + list);
    }
    cfg.method = method;
  }

  const std::string basis = common.getBasisSet().cStr();
  if (upper(basis) == "PLANEWAVE") {
    refuse("ORCA has no plane-wave basis");
  }
  if (!basis.empty()) {
    cfg.basis = basis;
  }
  if (common.getPlanewaveCutoffEv() > 0.0) {
    refuse("ORCA has no plane-wave cutoff");
  }
  for (auto k : common.getKMesh()) {
    if (k != 1) {
      refuse("ORCA is molecular, so kMesh must be empty or all ones");
    }
  }

  cfg.charge = common.getCharge();
  if (common.getSpinMultiplicity() < 0) {
    refuse("spinMultiplicity must be >= 0");
  }
  if (common.getSpinMultiplicity() > 0) {
    cfg.multiplicity = common.getSpinMultiplicity();
  }

  std::string scf;
  if (common.getScfEnergyToleranceEv() > 0.0) {
    scf += "  TolE " +
           number(common.getScfEnergyToleranceEv() / units::HARTREE_TO_EV) +
           "\n";
  }
  if (common.getScfMaxIterations() > 0) {
    scf += "  MaxIter " + std::to_string(common.getScfMaxIterations()) + "\n";
  }
  const auto smearing = common.getSmearing();
  switch (smearing.getKind()) {
  case ::CommonMethodSpec::Smearing::Kind::NONE:
    break;
  case ::CommonMethodSpec::Smearing::Kind::FERMI: {
    constexpr double kBoltzmannEvPerK = 8.617333262e-5;
    scf += "  SmearTemp " + number(smearing.getWidthEv() / kBoltzmannEvPerK) +
           "\n";
    break;
  }
  default:
    refuse("ORCA smears with a Fermi distribution only");
  }

  std::string keywords;
  std::string method_block;
  const std::string vdw = upper(common.getVanDerWaalsMethod().cStr());
  std::string s6_key;
  if (vdw == "DFT-D3") {
    keywords = "D3ZERO";
    s6_key = "D3S6";
  } else if (vdw == "DFT-D3(BJ)") {
    keywords = "D3BJ";
    s6_key = "D3S6";
  } else if (vdw == "DFT-D4") {
    keywords = "D4";
    s6_key = "D4S6";
  } else if (vdw == "DFT-D2") {
    keywords = "D2";
  } else if (!vdw.empty()) {
    refuse("no ORCA keyword for vanDerWaalsMethod " +
           std::string(common.getVanDerWaalsMethod().cStr()));
  }
  if (common.getVanDerWaalsS6() > 0.0) {
    if (s6_key.empty()) {
      refuse("vanDerWaalsS6 needs DFT-D3, DFT-D3(BJ) or DFT-D4");
    }
    method_block = "%method\n  " + s6_key + " " +
                   number(common.getVanDerWaalsS6()) + "\nend";
  }

  const std::string rel = upper(common.getRelativityMethod().cStr());
  if (rel == "ZORA") {
    append(keywords, "ZORA", " ");
  } else if (rel == "DKH" || rel == "DKH2") {
    append(keywords, "DKH2", " ");
  } else if (rel == "X2C") {
    append(keywords, "X2C", " ");
  } else if (!rel.empty()) {
    refuse("no ORCA keyword for relativityMethod " +
           std::string(common.getRelativityMethod().cStr()));
  }

  append(cfg.keywords, keywords, " ");
  if (!scf.empty()) {
    append(cfg.blocks, "%scf\n" + scf + "end", "\n");
  }
  append(cfg.blocks, method_block, "\n");
}

/// Lays the orca arm over cfg: a field that differs from its schema default
/// replaces cfg's, and keywords and blocks are appended.
inline void applyOrcaParams(const ::ORCAParams::Reader &params,
                            ORCAConfig &cfg) {
  using orca_detail::append;
  auto text = [](capnp::Text::Reader t) { return std::string(t.cStr()); };
  if (params.getMethod().size() > 0) {
    cfg.method = text(params.getMethod());
  }
  if (params.getBasis().size() > 0) {
    cfg.basis = text(params.getBasis());
  }
  append(cfg.keywords, text(params.getKeywords()), " ");
  append(cfg.blocks, text(params.getBlocks()), "\n");
  if (params.getCharge() != 0) {
    cfg.charge = params.getCharge();
  }
  if (params.getMultiplicity() != 1) {
    cfg.multiplicity = params.getMultiplicity();
  }
  if (params.getNprocs() != 1) {
    cfg.nprocs = params.getNprocs();
  }
  if (params.getMaxcoreMb() != 0) {
    cfg.maxcore_mb = params.getMaxcoreMb();
  }
  if (params.getExecutable().size() > 0) {
    cfg.executable = text(params.getExecutable());
  }
  if (params.getLibrary().size() > 0) {
    cfg.library = text(params.getLibrary());
  }
  cfg.guess_policy =
      params.getGuessPolicy() == ::ORCAParams::GuessPolicy::MOREAD
          ? ORCAConfig::GuessPolicy::MoRead
          : ORCAConfig::GuessPolicy::Fresh;
  if (params.getTimeoutS() != ORCAConfig::kDefaultTimeout) {
    cfg.timeout_s = params.getTimeoutS();
  }
  if (params.getScratchDir().size() > 0) {
    cfg.scratch_dir = text(params.getScratchDir());
  }
  if (params.getKeepFiles()) {
    cfg.keep_files = true;
  }
  if (!params.getFreshRetry()) {
    cfg.fresh_retry = false;
  }
}

/// The orca arm alone, as the engine plugin reads it.
[[nodiscard]] inline ORCAConfig
orcaConfigFrom(const ::ORCAParams::Reader &params) {
  ORCAConfig cfg;
  applyOrcaParams(params, cfg);
  return cfg;
}

/// The overlay lowered first, then the orca arm. cfg must hold the orca arm.
[[nodiscard]] inline ORCAConfig
orcaConfigFrom(const ::PotentialConfig::Reader &cfg) {
  ORCAConfig out;
  if (cfg.hasCommon()) {
    lowerCommonMethod(cfg.getCommon(), out);
  }
  applyOrcaParams(cfg.getOrca(), out);
  return out;
}

struct OrcaConfigureResult {
  bool ok = false;
  std::unique_ptr<ORCAPot> replacement;
  std::string message;
};

/// A replacement is handed back only when liborcac loaded and opened a
/// session; otherwise the running pot stays and the message says why.
[[nodiscard]] inline OrcaConfigureResult
configureOrca(const ::PotentialConfig::Reader &cfg) {
  OrcaConfigureResult result;
  if (cfg.which() == ::PotentialConfig::NONE) {
    if (cfg.hasCommon()) {
      result.message = "rgpot params common overlay needs the orca arm";
      return result;
    }
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  }
  if (cfg.which() != ::PotentialConfig::ORCA) {
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by ORCAPot (use matching backend pot)";
    return result;
  }
  try {
    auto pot = std::make_unique<ORCAPot>(orcaConfigFrom(cfg));
    if (!pot->available()) {
      result.message =
          "rgpot params orca arm not applied: " + pot->loadError();
      return result;
    }
    result.replacement = std::move(pot);
    result.ok = true;
    result.message = "rgpot params applied (orca arm)";
  } catch (const std::invalid_argument &error) {
    result.message =
        std::string("rgpot params orca arm rejected: ") + error.what();
  }
  return result;
}

} // namespace rgpot::rpc
