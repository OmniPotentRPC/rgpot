// MIT License — generic rgpot engine table for the ORCA subprocess.
// BuiltinParams carries the name "orca". `library`, when set, is the
// `orca` executable. Method, basis, charge and multiplicity stay on
// OrcaConfig's defaults unless a host uses the C++ class or the
// `rgpot_orca_*` entry.

#include "rgpot/OrcaPot/OrcaPot.hpp"
#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"

#include <exception>
#include <memory>
#include <string>

namespace rgpot::engine {

const char *backend_name() { return "orca"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin(config, config_len);
    if (!builtin.name.empty() && builtin.name != "orca") {
      out.error =
          "rgpot_engine_create(orca): unknown potential '" + builtin.name + "'";
      return out;
    }
    OrcaConfig c;
    if (!builtin.library.empty()) {
      c.executable = builtin.library;
    }
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
