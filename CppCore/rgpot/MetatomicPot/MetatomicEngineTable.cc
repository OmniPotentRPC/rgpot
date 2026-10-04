// MIT License — generic rgpot engine table for the Metatomic engine.
// The Metatomic-specific table (metatomic_c_abi.h) stays in
// MetatomicEngineAbi.cc; this file builds the potential from MetatomicParams
// for rgpot/engine/EngineTable.cc.

#include "rgpot/engine/EngineTable.hpp"
#include "rgpot/engine/TorchRuntime.hpp"
#include "rgpot/MetatomicPot/MetatomicPot.hpp"

#include <capnp/message.h>
#include <capnp/serialize.h>
#include <kj/io.h>

#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot::engine {

const char *backend_name() { return "metatomic"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const kj::ArrayPtr<const capnp::word> words(
        reinterpret_cast<const capnp::word *>(config),
        config_len / sizeof(capnp::word));
    capnp::FlatArrayMessageReader reader(words);
    const auto params = reader.getRoot<::MetatomicParams>();

    MetatomicConfig c;
    c.model_path = params.getModelPath().cStr();
    if (c.model_path.empty()) {
      out.error = "rgpot_engine_create(metatomic): modelPath required";
      return out;
    }
    c.device = params.getDevice().size() > 0 ? params.getDevice().cStr() : "cpu";
    c.extensions_directory = params.getExtensionsDirectory().cStr();
    c.check_consistency = params.getCheckConsistency();
    c.uncertainty_threshold = params.getUncertaintyThreshold();
    c.dtype_override = params.getDtypeOverride().cStr();
    c.n_symmetry_rotations = static_cast<long>(params.getNSymmetryRotations());
    c.random_rotation = params.getRandomRotation();
    c.so3_probe_scatter = params.getSo3ProbeScatter();
    c.torch_determinism =
        params.getTorchDeterminism() == ::MetatomicParams::TorchDeterminism::STRICT
            ? TorchDeterminismPolicy::Strict
            : TorchDeterminismPolicy::Fast;

    set_torch_threads(params.getIntraopThreads(), params.getInteropThreads());

    auto backend = std::make_unique<Backend>();
    auto pot = std::make_unique<MetatomicPot>(c);
    MetatomicPot *raw = pot.get();
    backend->pot = std::move(pot);
    backend->model_path = c.model_path;
    backend->set_charge_spin = [raw](int charge, int spin) {
      raw->setChargeSpin(charge, spin);
    };
    backend->set_num_threads = [](int intra_op, int inter_op) {
      set_torch_threads(intra_op, inter_op);
    };
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(metatomic): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
