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
    std::unique_ptr<PotentialBase> pot;
    const std::string &n = builtin.name;
    if (n == "lj") {
      pot = std::make_unique<LJPot>();
    } else if (n == "ljcluster") {
      pot = std::make_unique<LJClusterPot>();
    } else if (n == "morse") {
      pot = std::make_unique<MorsePot>();
    } else if (n == "zbl") {
      pot = std::make_unique<ZBLPot>();
#ifdef RGPOT_HAS_WATER
    } else if (n == "tip4p") {
      pot = std::make_unique<TIP4PPot>();
    } else if (n == "spce") {
      pot = std::make_unique<SPCEPot>();
    } else if (n == "tip4p_pt") {
      pot = std::make_unique<TIP4PPtPot>();
#endif
#ifdef RGPOT_HAS_FORTRAN_POTS
    } else if (n == "sw") {
      pot = std::make_unique<fortranpots::SWPot>();
    } else if (n == "edip") {
      pot = std::make_unique<fortranpots::EDIPPot>();
    } else if (n == "lenosky") {
      pot = std::make_unique<fortranpots::LenoskyPot>();
    } else if (n == "tersoff") {
      pot = std::make_unique<fortranpots::TersoffPot>();
    } else if (n == "eamal") {
      pot = std::make_unique<fortranpots::EAMAlPot>();
    } else if (n == "fehe") {
      pot = std::make_unique<fortranpots::FeHePot>();
    } else if (n == "cuh2") {
      pot = std::make_unique<fortranpots::CuH2Pot>();
    } else if (n == "waterh") {
      pot = std::make_unique<fortranpots::WaterHPot>();
#endif
    } else {
      out.error = "rgpot_engine_create(classical): unknown potential '" + n +
                  "'";
      return out;
    }
    auto backend = std::make_unique<Backend>();
    backend->pot = std::move(pot);
    out.backend = std::move(backend);
  } catch (const std::exception &e) {
    out.error = e.what();
  } catch (...) {
    out.error = "rgpot_engine_create(classical): unknown exception";
  }
  return out;
}

} // namespace rgpot::engine
