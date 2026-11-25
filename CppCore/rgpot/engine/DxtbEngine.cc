// MIT License
// Copyright 2023--present rgpot developers

#include "rgpot/DxtbPot/DxtbPot.hpp"
#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"

#include <algorithm>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <cctype>
#include <cmath>
#include <kj/io.h>

namespace rgpot::engine {
namespace {

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

DxtbConfig readConfig(const void *config, std::size_t config_len,
                      const Builtin &builtin) {
  const kj::ArrayPtr<const capnp::word> words(
      reinterpret_cast<const capnp::word *>(config),
      config_len / sizeof(capnp::word));
  capnp::FlatArrayMessageReader reader(words);
  const auto params = reader.getRoot<::BuiltinParams>();
  const auto native = params.getConfig();

  DxtbConfig out;
  out.python = builtin.library;
  switch (native.which()) {
  case ::PotentialConfig::NONE:
    return out;
  case ::PotentialConfig::XTB: {
    const auto xtb = native.getXtb();
    const std::string method = lower(xtb.getMethod().cStr());
    if (method == "gfn1" || method == "gfn1-xtb") {
      out.method = DxtbMethod::GFN1;
    } else if (method == "gfn2" || method == "gfn2-xtb") {
      out.method = DxtbMethod::GFN2;
    } else {
      throw DxtbError(1, "dxtb method must be gfn1 or gfn2");
    }
    out.accuracy = xtb.getAccuracy();
    out.electronic_temperature = xtb.getElectronicTemperature();
    out.max_iterations = xtb.getMaxIterations();
    out.charge = xtb.getCharge();
    out.uhf = xtb.getUhf();
    if (!std::isfinite(out.accuracy) || out.accuracy <= 0.0 ||
        !std::isfinite(out.electronic_temperature) ||
        out.electronic_temperature < 0.0 || out.max_iterations < 1 ||
        !std::isfinite(out.charge) || out.uhf < 0) {
      throw DxtbError(1, "invalid dxtb configuration");
    }
    return out;
  }
  default:
    throw DxtbError(1, "BuiltinParams.config must be none or xtb for dxtb");
  }
}

} // namespace

const char *backend_name() { return "dxtb"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin_identity(config, config_len);
    if (!builtin.name.empty() && builtin.name != "dxtb") {
      out.error =
          "rgpot_engine_create(dxtb): unknown potential '" + builtin.name + "'";
      return out;
    }
    const DxtbConfig cfg = readConfig(config, config_len, builtin);
    auto backend = std::make_unique<Backend>();
    auto pot = std::make_unique<DxtbPot>(cfg);
    DxtbPot *raw = pot.get();
    backend->pot = std::move(pot);
    backend->set_charge_spin = [raw](int charge, int multiplicity) {
      raw->setChargeSpin(charge, multiplicity);
    };
    backend->set_num_threads = [raw](int intra_op, int inter_op) {
      raw->setNumThreads(intra_op, inter_op);
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(dxtb): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
