// MIT License — generic rgpot engine table for the classical potentials.
// BuiltinParams names the potential: lj, ljcluster, morse, zbl and, when the
// water option is on, tip4p, spce, tip4p_pt. When the Fortran kernels are
// built: sw, edip, lenosky, tersoff, eamal, fehe, cuh2, waterh. Every
// potential takes its default parameters.

#include "rgpot/engine/BuiltinParams.hpp"
#include "rgpot/engine/EngineTable.hpp"

#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#ifdef RGPOT_HAS_WATER
#include "rgpot/Water/WaterPots.hpp"
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif

namespace rgpot::engine {

const char *backend_name() { return "classical"; }

Created create(const void *config, std::size_t config_len) {
  Created out;
  try {
    const Builtin builtin = read_builtin(config, config_len);
    std::unique_ptr<Backend> backend;
    const std::string &n = builtin.name;
    if (n == "lj") {
      backend = backend_for(std::make_unique<LJPot>());
    } else if (n == "ljcluster") {
      backend = backend_for(std::make_unique<LJClusterPot>());
    } else if (n == "morse") {
      backend = backend_for(std::make_unique<MorsePot>());
    } else if (n == "zbl") {
      backend = backend_for(std::make_unique<ZBLPot>());
#ifdef RGPOT_HAS_WATER
    } else if (n == "tip4p") {
      backend = backend_for(std::make_unique<TIP4PPot>());
    } else if (n == "spce") {
      backend = backend_for(std::make_unique<SPCEPot>());
    } else if (n == "tip4p_pt") {
      backend = backend_for(std::make_unique<TIP4PPtPot>());
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
    } else if (n == "sw") {
      backend = backend_for(std::make_unique<fortranpots::SWPot>());
    } else if (n == "edip") {
      backend = backend_for(std::make_unique<fortranpots::EDIPPot>());
    } else if (n == "lenosky") {
      backend = backend_for(std::make_unique<fortranpots::LenoskyPot>());
    } else if (n == "tersoff") {
      // All-silicon stays on the Fortran gather. Any other Z is the
      // 1989 kernel, inside TersoffPot::forceImpl.
      backend = backend_for(std::make_unique<fortranpots::TersoffPot>());
    } else if (n == "eamal") {
      backend = backend_for(std::make_unique<fortranpots::EAMAlPot>());
    } else if (n == "fehe") {
      backend = backend_for(std::make_unique<fortranpots::FeHePot>());
    } else if (n == "cuh2") {
      backend = backend_for(std::make_unique<fortranpots::CuH2Pot>());
    } else if (n == "waterh") {
      backend = backend_for(std::make_unique<fortranpots::WaterHPot>());
#endif
    } else {
      out.error = "rgpot_engine_create(classical): unknown potential '" + n +
                  "'";
      return out;
    }
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(classical): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
