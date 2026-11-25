// MIT License
// Copyright 2023--present rgpot developers
//
// Generic rgpot engine table for ORCA (liborcac through ORCAPot).
// The configuration is the ORCAParams message; library names liborcac.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/rpc/OrcaConfigure.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include <string>

namespace rgpot::engine {

const char *backend_name() { return "orca"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    auto pot = std::make_unique<ORCAPot>(
        rpc::orcaConfigFrom(reader.getRoot<::ORCAParams>()));
    auto *raw = pot.get();
    auto backend = std::make_unique<Backend>();
    backend->pot = std::move(pot);
    backend->set_charge_spin = [raw](int charge, int spin) {
      if (spin < 1) {
        throw Unsupported("orca spin multiplicity must be >= 1; got " +
                          std::to_string(spin));
      }
      if (!raw->available()) {
        throw Unsupported(raw->loadError());
      }
      raw->setChargeMultiplicity(charge, spin);
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(orca): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
