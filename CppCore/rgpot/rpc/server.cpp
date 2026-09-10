// MIT License
// Copyright 2023--present rgpot developers

/**
 * @brief Implementation of the standalone Cap'n Proto potential server.
 *
 * This file implements a basic RPC server which exposes toy potentials over a
 * network interface. It utilizes the @c EzRpcServer for handling requests.
 */

#include <capnp/ez-rpc.h>
#include <capnp/message.h>
#include <kj/debug.h>

#ifdef RGPOT_HAS_FORTRAN_POTS
#include "rgpot/fortran/FortranPots.hpp"
#endif // RGPOT_HAS_FORTRAN_POTS

#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif // RGPOT_HAS_XTB

#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif // RGPOT_HAS_TBLITE

#ifdef RGPOT_HAS_DFTD3
#include "rgpot/D3Pot/D3Pot.hpp"
#endif // RGPOT_HAS_DFTD3

#ifdef RGPOT_HAS_DFTD4
#include "rgpot/D4Pot/D4Pot.hpp"
#endif // RGPOT_HAS_DFTD4

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#endif // RGPOT_HAS_METATOMIC

#ifdef RGPOT_HAS_UMA
#include "rgpot/UmaPot/UmaPot.hpp"
#endif
#ifdef RGPOT_HAS_SKALA
#include "rgpot/SkalaPot/SkalaPot.hpp"
#endif // RGPOT_HAS_UMA

#include "rgpot/CPMDPot/CPMDPot.hpp"
#include "rgpot/LennardJones/LJClusterPot.hpp"
#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Morse/MorsePot.hpp"
#include "rgpot/NWChemPot/NWChemPot.hpp"
#include "rgpot/Potential.hpp"
#include "rgpot/ZBL/ZBLPot.hpp"
#include "rgpot/types/AtomMatrix.hpp"
#include "rgpot/types/adapters/capnp/capnp_adapter.hpp"
#include "rgpot/units.hpp"

/**
 * @class GenericPotImpl
 * @brief Server implementation for the Potential RPC interface.
 *
 * This class wraps a polymorphic @c PotentialBase instance and dispatches
 * RPC calculate requests to the underlying physics engine.
 */
class GenericPotImpl final : public Potential::Server {
private:
  std::unique_ptr<rgpot::PotentialBase>
      m_potential; //!< The polymorphic potential engine.
  /// Optional typed handle when backend is NWChem (for configure()).
  rgpot::NWChemPot *m_nwchem = nullptr;
  /// Optional typed handle when backend is CPMD (for configure()).
  rgpot::CPMDPot *m_cpmd = nullptr;

public:
  /**
   * @brief Constructor for GenericPotImpl.
   * @param pot Ownership of a PotentialBase derived object.
   */
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)) {}

  /**
   * @brief Constructor retaining an NWChemPot pointer for configure().
   * Members initialize in declaration order (m_potential then m_nwchem).
   */
  GenericPotImpl(std::unique_ptr<rgpot::NWChemPot> pot)
      : m_potential(std::move(pot)),
        m_nwchem(static_cast<rgpot::NWChemPot *>(m_potential.get())) {}

  GenericPotImpl(std::unique_ptr<rgpot::CPMDPot> pot)
      : m_potential(std::move(pot)),
        m_cpmd(static_cast<rgpot::CPMDPot *>(m_potential.get())) {}

  /**
   * @details
   * This method performs the following translation steps:
   * 1. Extracts the @c ForceInput (fip) from the RPC context.
   * 2. Validates the size of the atomic number list.
   * 3. Converts Cap'n Proto lists into native @c AtomMatrix and @c std::vector
   * types.
   * 4. Executes the calculation via the @c PotentialBase virtual operator.
   * 5. Populates the @c PotentialResult with energy and force data.
   *
   * @param context The Cap'n Proto RPC call context.
   * @return An asynchronous promise for completion.
   */
  kj::Promise<void> calculate(CalculateContext context) override {
    auto fip = context.getParams().getFip();
    const size_t numAtoms = fip.getPos().size() / 3;

    KJ_REQUIRE(fip.getAtmnrs().size() == numAtoms, "AtomNumbers size mismatch");

    // Unit conversion factors (caller units -> internal angstrom/eV)
    std::string lengthUnit = fip.getLengthUnit();
    std::string energyUnit = fip.getEnergyUnit();
    double len_to_angstrom = rgpot::units::unit_conversion_factor(lengthUnit, "angstrom");
    double ev_to_caller = rgpot::units::unit_conversion_factor("eV", energyUnit);
    // Force unit: energy/length in caller's system
    double force_to_caller = ev_to_caller / len_to_angstrom;

    rgpot::types::AtomMatrix nativePositions =
        rgpot::types::adapt::capnp::convertPositionsFromCapnp(fip.getPos(),
                                                              numAtoms);
    // Convert positions from caller's length unit to angstrom
    if (len_to_angstrom != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        nativePositions.data()[i] *= len_to_angstrom;
      }
    }

    std::vector<int> nativeAtomTypes =
        rgpot::types::adapt::capnp::convertAtomNumbersFromCapnp(
            fip.getAtmnrs());
    std::array<std::array<double, 3>, 3> nativeBoxMatrix =
        rgpot::types::adapt::capnp::convertBoxMatrixFromCapnp(fip.getBox());
    // Convert box from caller's length unit to angstrom
    if (len_to_angstrom != 1.0) {
      for (auto &row : nativeBoxMatrix)
        for (auto &val : row)
          val *= len_to_angstrom;
    }

    // Potential always computes in eV/angstrom
    auto [energy, forces, variance] =
        (*m_potential)(nativePositions, nativeAtomTypes, nativeBoxMatrix);
    (void)variance;

    // Convert results to caller's units
    auto result = context.getResults();
    auto pres = result.initResult();
    pres.setEnergy(energy * ev_to_caller);

    auto forcesList = pres.initForces(numAtoms * 3);
    if (force_to_caller != 1.0) {
      for (size_t i = 0; i < numAtoms * 3; ++i) {
        forces.data()[i] *= force_to_caller;
      }
    }
    rgpot::types::adapt::capnp::populateForcesToCapnp(forcesList, forces);

    return kj::READY_NOW;
  }

  kj::Promise<void> configure(ConfigureContext context) override {
    auto cfg = context.getParams().getConfig();
    auto results = context.getResults();
    if (!m_nwchem && !m_cpmd) {
      results.setOk(false);
      results.setMessage("configure() only supported for NWChem or CPMD backend");
      return kj::READY_NOW;
    }
    std::string msg;
    bool ok = m_nwchem ? m_nwchem->setPotentialConfig(cfg, &msg)
                       : m_cpmd->setPotentialConfig(cfg, &msg);
    results.setOk(ok);
    results.setMessage(msg);
    return kj::READY_NOW;
  }
};

/**
 * @details
 * The main entry point handles command-line arguments to specify the
 * network port and the potential type. It instantiates the requested
 * physics engine and blocks until the server is terminated.
 *
 * # Usage
 * @c ./potserv <port> <PotentialType>
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, 1 on initialization failure.
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <port> <PotentialType>" << std::endl;
    std::cerr << "  Available PotentialTypes: CuH2, LJ, LJCluster, Morse, ZBL"
#ifdef RGPOT_HAS_XTB
              << ", XTB, GFNFF, GFN0xTB, GFN1xTB"
#endif
#ifdef RGPOT_HAS_TBLITE
              << ", TBLite, TBLiteGFN1, TBLiteIPEA1"
#endif
#ifdef RGPOT_HAS_DFTD3
              << ", D3"
#endif
#ifdef RGPOT_HAS_DFTD4
              << ", D4"
#endif
#ifdef RGPOT_HAS_METATOMIC
              << ", Metatomic:<model_path>"
#endif
#ifdef RGPOT_HAS_UMA
              << ", Uma:<model_path>, Uma:<model_path>:<task>"
#endif
#ifdef RGPOT_HAS_SKALA
              << ", Skala, Skala:<basis>"
#endif
              << ", NWChem"
              << ", CPMD"
              << std::endl;
    return 1;
  }

  int port = 12345;
  try {
    port = std::stoi(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Invalid port argument '" << argv[1]
              << "'. Using default 12345." << std::endl;
  }

  std::string pot_type = argv[2];
  std::unique_ptr<rgpot::PotentialBase> potential_to_use;

#ifdef RGPOT_HAS_FORTRAN_POTS
  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::fortranpots::CuH2Pot>();
  } else
#endif // RGPOT_HAS_FORTRAN_POTS
      if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
  } else if (pot_type == "LJCluster") {
    std::cout << "Loading LJ cluster potential (free boundaries)..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::LJClusterPot>();
  } else if (pot_type == "Morse") {
    std::cout << "Loading Morse potential (Pt parameters)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::MorsePot>();
  } else if (pot_type == "ZBL") {
    std::cout << "Loading ZBL potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::ZBLPot>();
#ifdef RGPOT_HAS_XTB
  } else if (pot_type == "XTB") {
    std::cout << "Loading XTB potential (GFN2-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>();
  } else if (pot_type == "GFNFF") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFNFF;
    std::cout << "Loading XTB potential (GFNFF)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN1xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN1xTB;
    std::cout << "Loading XTB potential (GFN1-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
  } else if (pot_type == "GFN0xTB") {
    rgpot::XTBConfig cfg;
    cfg.method = rgpot::GFNMethod::GFN0xTB;
    std::cout << "Loading XTB potential (GFN0-xTB)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::XTBPot>(cfg);
#endif // RGPOT_HAS_XTB
#ifdef RGPOT_HAS_TBLITE
  } else if (pot_type == "TBLite" || pot_type == "TBLiteGFN2") {
    std::cout << "Loading TBLite potential (GFN2)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>();
  } else if (pot_type == "TBLiteGFN1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::GFN1;
    std::cout << "Loading TBLite potential (GFN1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
  } else if (pot_type == "TBLiteIPEA1") {
    rgpot::TBLiteConfig cfg;
    cfg.method = rgpot::TBLiteMethod::IPEA1;
    std::cout << "Loading TBLite potential (IPEA1)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::TBLitePot>(cfg);
#endif // RGPOT_HAS_TBLITE
#ifdef RGPOT_HAS_DFTD3
  } else if (pot_type == "D3") {
    std::cout << "Loading D3 potential (BJ PBE ATM)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D3Pot>();
#endif // RGPOT_HAS_DFTD3
#ifdef RGPOT_HAS_DFTD4
  } else if (pot_type == "D4") {
    std::cout << "Loading D4 potential (PBE ATM charge 0)..." << std::endl;
    potential_to_use = std::make_unique<rgpot::D4Pot>();
#endif // RGPOT_HAS_DFTD4
#ifdef RGPOT_HAS_METATOMIC
  } else if (pot_type.rfind("Metatomic:", 0) == 0) {
    auto model_path = pot_type.substr(10);
    rgpot::MetatomicConfig cfg;
    cfg.model_path = model_path;
    std::cout << "Loading Metatomic potential from '" << model_path << "'..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::MetatomicPot>(cfg);
#endif // RGPOT_HAS_METATOMIC
#ifdef RGPOT_HAS_UMA
  } else if (pot_type.rfind("Uma:", 0) == 0) {
    rgpot::UmaConfig cfg;
    auto rest = pot_type.substr(4);
    const auto colon = rest.rfind(':');
    if (colon != std::string::npos) {
      const auto tail = rest.substr(colon + 1);
      if (tail == "omol" || tail == "omat" || tail == "oc20" ||
          tail == "oc22" || tail == "oc25" || tail == "odac" ||
          tail == "omc") {
        cfg.task_name = tail;
        rest = rest.substr(0, colon);
      }
    }
    cfg.model_path = rest;
    std::cout << "Loading UMA AOTI package '" << cfg.model_path
              << "' task='" << cfg.task_name << "'..." << std::endl;
    potential_to_use = std::make_unique<rgpot::UmaPot>(cfg);
#endif // RGPOT_HAS_UMA
#ifdef RGPOT_HAS_SKALA
  } else if (pot_type == "Skala" || pot_type.rfind("Skala:", 0) == 0) {
    rgpot::SkalaConfig cfg;
    if (pot_type.rfind("Skala:", 0) == 0)
      cfg.basis = pot_type.substr(6);
    std::cout << "Loading Skala XC via NWChem DFT (xc=" << cfg.xc
              << " basis=" << cfg.basis << ")..." << std::endl;
    potential_to_use = std::make_unique<rgpot::SkalaPot>(cfg);
#endif // RGPOT_HAS_SKALA
  } else if (pot_type == "NWChem") {
    std::cout << "Loading NWChem potential (dlopen libnwchemc)..."
              << std::endl;
    auto nw = std::make_unique<rgpot::NWChemPot>();
    if (!nw->available()) {
      std::cerr << "Warning: libnwchemc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr; // use dedicated path below
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(nw)),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else if (pot_type == "CPMD") {
    std::cout << "Loading CPMD potential (dlopen libcpmdc)..." << std::endl;
    auto cp = std::make_unique<rgpot::CPMDPot>();
    if (!cp->available()) {
      std::cerr << "Warning: libcpmdc not loaded; calculate() will fail "
                   "until engine is available (configure() still accepted)."
                << std::endl;
    }
    potential_to_use = nullptr;
    capnp::EzRpcServer server(kj::heap<GenericPotImpl>(std::move(cp)),
                              "localhost", port);
    auto &waitScope = server.getWaitScope();
    std::cout << "Server running on port " << port << " with " << pot_type
              << " potential." << std::endl;
    kj::NEVER_DONE.wait(waitScope);
    return 0;
  } else {
    std::cerr << "Error: Unknown potential type '" << pot_type << "'"
              << std::endl;
    return 1;
  }

  if (!potential_to_use) {
    std::cerr << "Error: potential not initialized" << std::endl;
    return 1;
  }

  capnp::EzRpcServer server(
      kj::heap<GenericPotImpl>(std::move(potential_to_use)), "localhost", port);

  auto &waitScope = server.getWaitScope();
  std::cout << "Server running on port " << port << " with " << pot_type
            << " potential." << std::endl;
  kj::NEVER_DONE.wait(waitScope);

  return 0;
}
