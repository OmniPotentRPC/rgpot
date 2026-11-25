#pragma once
// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief One construction entry point: the potential a PotentialConfig
 * names.
 *
 * makePotential dispatches on the union arm of a PotentialConfig and returns
 * the potential that arm configures. The arm's own parameters decide
 * everything; the shared fields (common, commonPatch, neighborList,
 * rpcEndpoint, logPotentialCalls, emtRasmussenParameters) are not read,
 * matching the setPotentialConfig methods and the rpc/ *Configure helpers,
 * none of which reads them either. The orca arm is the exception: liborcac
 * receives the whole PotentialConfig and lowers common.
 *
 * In-process arms: lj, morse, zbl, d2, d2cpmd, mullerBrown, composite (its
 * PotentialSpec terms as rgpot::rpc::potentialFromSpec builds them), cpmd and
 * nwchem (their frontends, which dlopen the split cpmdc and nwchemc engines),
 * mopac (MOPACPot), orca (ORCACPot over the split liborcac shim, created
 * from the whole PotentialConfig so liborcac lowers common itself; library
 * from ORCAC_LIBRARY, RGPOT_ORCAC_ENGINE, liborcac.so.1, liborcac.so),
 * lammps (the lammpc profile of LammpsPot), and, when the
 * build has them, d3 (-Dwith_dftd3), d4 (-Dwith_dftd4), xtb (-Dwith_xtb),
 * tblite (-Dwith_tblite), skala (-Dwith_skala), cp2k (-Dwith_cp2k) and expr
 * (-Dwith_expr).
 *
 * Engine-library arms, built as an rgpot::EnginePot over the generic engine
 * C ABI (rgpot/engine_c_abi.h) with the arm's params as the configuration
 * message: metatomic (libmetatomic_engine; MetatomicParams.enginePath,
 * RGPOT_METATOMIC_ENGINE, METATOMIC_ENGINE), uma (libuma_engine;
 * RGPOT_UMA_ENGINE), xtb in a build without -Dwith_xtb (libxtb_engine;
 * RGPOT_XTB_ENGINE, XTB_ENGINE) and plugin
 * (PluginPotentialParams.libraryPath, else lib<pluginName>_engine through
 * RGPOT_<PLUGINNAME>_ENGINE; the message is opaquePayload). Bare library
 * names also search EON_POTENTIALS_PATH and RGPOT_ENGINE_PATH.
 *
 * An arm this build cannot construct, and an arm newer than this build's
 * schema, throw std::invalid_argument naming the arm and, where one exists,
 * the build option that enables it. A backend that is built in but cannot
 * load (a missing engine library, an unavailable runtime) throws
 * std::runtime_error from its own constructor.
 */

#include <cstddef>
#include <memory>

#include "rgpot/Potential.hpp"
#include "rgpot/rpc/Potentials.capnp.h"

namespace rgpot {

/// The potential @p config names. See the file comment for the arms.
[[nodiscard]] std::unique_ptr<PotentialBase>
makePotential(const ::PotentialConfig::Reader &config);

/// The potential a PotentialConfig flat-array message names: @p bytes is
/// what capnp::messageToFlatArray wrote, @p size bytes long (a multiple of
/// eight). Throws std::invalid_argument for bytes that are not such a
/// message.
[[nodiscard]] std::unique_ptr<PotentialBase>
makePotential(const void *bytes, std::size_t size);

} // namespace rgpot
