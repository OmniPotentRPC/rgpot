// MIT License — generic rgpot engine table for NWChem (libnwchemc through the
// profile ABI). The configuration is the NWChemParams message.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

namespace rgpot::engine {

const char *backend_name() { return "nwchem"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    auto backend = std::make_unique<Backend>();
    backend->pot = std::make_unique<NWChemPot>(reader.getRoot<::NWChemParams>());
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(nwchem): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
