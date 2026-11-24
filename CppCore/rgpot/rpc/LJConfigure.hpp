#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief PotentialConfig.lj for an LJPot or LJClusterPot backend.
 *
 * LJPot and LJClusterPot fix their parameters at construction, so applying
 * the lj arm builds a replacement pot. LJClusterPot has free boundaries and
 * refuses tailCorrection = true.
 */

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::rpc {

[[nodiscard]] inline LJConfig
ljConfigFrom(const ::LennardJonesParams::Reader &params, LJConfig base = {}) {
  base.u0 = params.getU0();
  base.psi = params.getPsi();
  base.cutoff = params.getCutoff();
  base.tail_correction = params.getTailCorrection();
  return base;
}

[[nodiscard]] inline LJClusterConfig
ljClusterConfigFrom(const ::LennardJonesParams::Reader &params,
                    LJClusterConfig base = {}) {
  if (params.getTailCorrection()) {
    throw std::invalid_argument("LJClusterPot has free boundaries and does not "
                                "support tail_correction");
  }
  base.u0 = params.getU0();
  base.psi = params.getPsi();
  base.cutoff = params.getCutoff();
  return base;
}

struct LJConfigureResult {
  bool ok = false;
  /// Set when the lj arm was accepted; the caller swaps it in.
  std::unique_ptr<LJPot> replacement;
  std::string message;
};

/// none is a no-op, lj rebuilds the pot, every other arm is refused.
[[nodiscard]] inline LJConfigureResult
configureLJ(const ::PotentialConfig::Reader &cfg, const LJPot &current) {
  LJConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::LJ: {
    const LJConfig next = ljConfigFrom(cfg.getLj(), current.config());
    if (!std::isfinite(next.u0) || !std::isfinite(next.psi) ||
        !std::isfinite(next.cutoff) || !(next.u0 > 0.0) || !(next.psi > 0.0) ||
        !(next.cutoff > 0.0)) {
      result.message =
          "rgpot params lj arm rejected: u0, psi and cutoff must be "
          "finite and positive";
      return result;
    }
    std::ostringstream summary;
    summary << "u0=" << next.u0 << " psi=" << next.psi
            << " cutoff=" << next.cutoff
            << " tail_correction=" << (next.tail_correction ? "true" : "false");
    try {
      result.replacement = std::make_unique<LJPot>(next);
    } catch (const std::invalid_argument &error) {
      result.message = "rgpot params lj arm rejected (" + summary.str() +
                       "): " + error.what();
      return result;
    }
    result.ok = true;
    result.message = "rgpot params applied (lj arm): " + summary.str();
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by LJPot (use matching backend pot)";
    return result;
  }
}

struct LJClusterConfigureResult {
  bool ok = false;
  /// Set when the lj arm was accepted; the caller swaps it in.
  std::unique_ptr<LJClusterPot> replacement;
  std::string message;
};

/// none is a no-op, lj rebuilds the pot (refusing tailCorrection), every
/// other arm is refused.
[[nodiscard]] inline LJClusterConfigureResult
configureLJCluster(const ::PotentialConfig::Reader &cfg,
                   const LJClusterPot &current) {
  LJClusterConfigureResult result;
  switch (cfg.which()) {
  case ::PotentialConfig::NONE:
    result.ok = true;
    result.message = "no-op (rgpot params: none)";
    return result;
  case ::PotentialConfig::LJ: {
    const auto params = cfg.getLj();
    if (!std::isfinite(params.getU0()) || !std::isfinite(params.getPsi()) ||
        !std::isfinite(params.getCutoff()) || !(params.getU0() > 0.0) ||
        !(params.getPsi() > 0.0) || !(params.getCutoff() > 0.0)) {
      result.message =
          "rgpot params lj arm rejected: u0, psi and cutoff must be "
          "finite and positive";
      return result;
    }
    std::ostringstream summary;
    summary << "u0=" << params.getU0() << " psi=" << params.getPsi()
            << " cutoff=" << params.getCutoff() << " tail_correction="
            << (params.getTailCorrection() ? "true" : "false");
    try {
      const LJClusterConfig next =
          ljClusterConfigFrom(params, current.config());
      result.replacement = std::make_unique<LJClusterPot>(next);
    } catch (const std::invalid_argument &error) {
      result.message = "rgpot params lj arm rejected (" + summary.str() +
                       "): " + error.what();
      return result;
    }
    result.ok = true;
    result.message = "rgpot params applied (lj arm): " + summary.str();
    return result;
  }
  default:
    result.message = "rgpot params arm " + potentialConfigArm(cfg) +
                     " not handled by LJClusterPot (use matching backend pot)";
    return result;
  }
}

} // namespace rgpot::rpc
