// MIT License — generic rgpot engine table for ORCA. The configuration is
// the OrcaParams message. The executable is OrcaParams.executable, else
// ORCA_EXE, else an orca binary on PATH.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/ORCAPot/ORCAPot.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

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
    ORCAPot::Config cfg;
    cfg.method = params.getMethod();
    cfg.basis = params.getBasis();
    cfg.charge = params.getCharge();
    cfg.multiplicity = params.getMultiplicity();
    cfg.keywords = params.getKeywords();
    cfg.executable = params.getExecutable();
    auto orca = std::make_unique<ORCAPot>(cfg);
    ORCAPot *raw = orca.get();
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
