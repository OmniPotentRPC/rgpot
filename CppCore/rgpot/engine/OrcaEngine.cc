// MIT License — generic rgpot engine table for the ORCA subprocess.
// The configuration is the OrcaParams message. `executable` empty means
// RGPOT_ORCA, then ORCA, then ORCA_EXE, then `orca` on PATH.

#include "rgpot/OrcaPot/OrcaPot.hpp"
#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include <exception>
#include <memory>
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
    const auto params = reader.getRoot<::OrcaParams>();
    OrcaConfig c;
    c.method = params.getMethod();
    c.basis = params.getBasis();
    c.charge = params.getCharge();
    c.multiplicity = params.getMultiplicity();
    c.keywords = params.getKeywords();
    c.executable = params.getExecutable();
    auto orca = std::make_unique<OrcaPot>(c);
    OrcaPot *raw = orca.get();
    auto backend = std::make_unique<Backend>();
    backend->pot = std::move(orca);
    backend->set_charge_spin = [raw](int charge, int spin) {
      raw->setChargeSpin(charge, spin);
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
