#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @file BuiltinParams.hpp
 * @brief Reads the `BuiltinParams` arm (a name and an optional library path)
 * that selects a backend with no parameters of its own.
 */

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Models/MullerBrownConfig.hpp"
#include "rgpot/XTBPot/XTBConfig.hpp"
#include "rgpot/rpc/PotentialConfigArm.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::engine {

struct Builtin {
  std::string name;    ///< Lowercased.
  std::string library; ///< Optional explicit engine library path.
  std::optional<rgpot::XTBConfig> xtb; ///< Present for the XTB config arm.
  std::optional<rgpot::LJConfig> lj; ///< Present for the LJ config arm on "lj".
  std::optional<rgpot::LJClusterConfig>
      ljcluster; ///< Present for the LJ config arm on "ljcluster".
  /// Present for the Muller-Brown config arm. Empty lists keep the published
  /// defaults carried by MullerBrownConfig.
  std::optional<rgpot::MullerBrownConfig> mullerBrown;
};

inline Builtin read_builtin(const void *config, std::size_t config_len) {
  const kj::ArrayPtr<const capnp::word> words(
      reinterpret_cast<const capnp::word *>(config),
      config_len / sizeof(capnp::word));
  capnp::FlatArrayMessageReader reader(words);
  const auto params = reader.getRoot<::BuiltinParams>();
  Builtin out;
  out.name = params.getName().cStr();
  std::transform(out.name.begin(), out.name.end(), out.name.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  out.library = params.getLibrary().cStr();
  const auto native = params.getConfig();
  if (native.which() == ::PotentialConfig::XTB) {
    const auto source = native.getXtb();
    rgpot::XTBConfig target;
    std::string method = source.getMethod().cStr();
    std::transform(method.begin(), method.end(), method.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (method == "gfnff") {
      target.method = rgpot::GFNMethod::GFNFF;
    } else if (method == "gfn0") {
      target.method = rgpot::GFNMethod::GFN0xTB;
    } else if (method == "gfn1") {
      target.method = rgpot::GFNMethod::GFN1xTB;
    } else if (method == "gfn2") {
      target.method = rgpot::GFNMethod::GFN2xTB;
    } else {
      throw std::invalid_argument("BuiltinParams.config.xtb: unknown method '" +
                                  method + "'");
    }
    target.accuracy = source.getAccuracy();
    target.electronic_temperature = source.getElectronicTemperature();
    target.max_iterations = source.getMaxIterations();
    target.charge = source.getCharge();
    target.uhf = source.getUhf();
    out.xtb = target;
  } else if (native.which() == ::PotentialConfig::LJ) {
    const auto source = native.getLj();
    const double u0 = source.getU0();
    const double psi = source.getPsi();
    const double cutoff = source.getCutoff();
    const bool tail_correction = source.getTailCorrection();
    if (!std::isfinite(u0) || !std::isfinite(psi) || !std::isfinite(cutoff) ||
        !(u0 > 0.0) || !(psi > 0.0) || !(cutoff > 0.0)) {
      throw std::invalid_argument("BuiltinParams.config.lj: u0, psi and cutoff "
                                  "must be finite and positive");
    }
    if (out.name == "ljcluster") {
      if (tail_correction) {
        throw std::invalid_argument(
            "BuiltinParams.config.lj: ljcluster has free boundaries and does "
            "not support tail_correction");
      }
      rgpot::LJClusterConfig target;
      target.u0 = u0;
      target.psi = psi;
      target.cutoff = cutoff;
      out.ljcluster = target;
    } else if (out.name == "lj" || out.name.empty()) {
      rgpot::LJConfig target;
      target.u0 = u0;
      target.psi = psi;
      target.cutoff = cutoff;
      target.tail_correction = tail_correction;
      out.lj = target;
    } else {
      throw std::invalid_argument("BuiltinParams.config.lj: arm lj is only "
                                  "valid for 'lj' or 'ljcluster', not '" +
                                  out.name + "'");
    }
  } else if (native.which() == ::PotentialConfig::MULLER_BROWN) {
    const auto wire = native.getMullerBrown();
    out.mullerBrown = rgpot::mullerBrownConfigFromLists(
        wire.getAmplitude(), wire.getA(), wire.getB(), wire.getC(),
        wire.getX0(), wire.getY0());
  } else if (native.which() != ::PotentialConfig::NONE) {
    throw std::invalid_argument("BuiltinParams.config arm " +
                                rgpot::rpc::potentialConfigArm(native) +
                                " does not match a builtin potential");
  }
  return out;
}

} // namespace rgpot::engine
