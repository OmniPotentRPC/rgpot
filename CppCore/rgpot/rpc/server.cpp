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

#ifdef RGPOT_HAS_FORTRAN
#include "rgpot/CuH2/CuH2Pot.hpp"
#endif // RGPOT_HAS_FORTRAN

#ifdef RGPOT_HAS_XTB
#include "rgpot/XTBPot/XTBPot.hpp"
#endif // RGPOT_HAS_XTB

#ifdef RGPOT_HAS_TBLITE
#include "rgpot/TBLitePot/TBLitePot.hpp"
#endif // RGPOT_HAS_TBLITE

#ifdef RGPOT_HAS_METATOMIC
#include "rgpot/MetatomicPot/MetatomicPot.hpp"
#endif // RGPOT_HAS_METATOMIC

#include "rgpot/LennardJones/LJPot.hpp"
#include "rgpot/Potential.hpp"
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

public:
  /**
   * @brief Constructor for GenericPotImpl.
   * @param pot Ownership of a PotentialBase derived object.
   */
  GenericPotImpl(std::unique_ptr<rgpot::PotentialBase> pot)
      : m_potential(std::move(pot)) {}

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
    auto [energy, forces] =
        (*m_potential)(nativePositions, nativeAtomTypes, nativeBoxMatrix);

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
    std::cerr << "  Available PotentialTypes: CuH2, LJ"
#ifdef RGPOT_HAS_XTB
              << ", XTB, GFNFF, GFN0xTB, GFN1xTB"
#endif
#ifdef RGPOT_HAS_TBLITE
              << ", TBLite, TBLiteGFN1, TBLiteIPEA1"
#endif
#ifdef RGPOT_HAS_METATOMIC
              << ", Metatomic:<model_path>"
#endif
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

  if (pot_type == "CuH2") {
    std::cout << "Loading CuH2 potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::CuH2Pot>();
  } else if (pot_type == "LJ") {
    std::cout << "Loading LJ potential..." << std::endl;
    potential_to_use = std::make_unique<rgpot::LJPot>();
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
#ifdef RGPOT_HAS_METATOMIC
  } else if (pot_type.rfind("Metatomic:", 0) == 0) {
    auto model_path = pot_type.substr(10);
    rgpot::MetatomicConfig cfg;
    cfg.model_path = model_path;
    std::cout << "Loading Metatomic potential from '" << model_path << "'..."
              << std::endl;
    potential_to_use = std::make_unique<rgpot::MetatomicPot>(cfg);
#endif // RGPOT_HAS_METATOMIC
  } else {
    std::cerr << "Error: Unknown potential type '" << pot_type << "'"
              << std::endl;
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
